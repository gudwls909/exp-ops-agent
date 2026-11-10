"""
메모리(runbook) ablation: 검증된 조치를 기억하면 같은 유형의 장애를 더 빨리, 더 싸게 해결하나

설계 (논문 ablation과 같다: 조건 하나만 바꾸고 나머지는 동일)
- 조건 A (memory off): OPS_MEMORY=0 → find_runbooks가 항상 빈 결과. 도구 목록·프롬프트는 똑같다
- 조건 B (memory on) : 에피소드가 지나며 "결과가 확인된 조치"가 runbook으로 쌓인다
- 에피소드마다 같은 "유형"이지만 숫자가 다른 장애를 띄운다
    1) shard 멈춤: 특정 shard를 읽으면 멈춘다(재시작해도 또 멈춤). shard 번호는 에피소드마다 랜덤
    2) OOM: 2단계 batch가 너무 큼. batch 크기는 에피소드마다 다름
    3) 정상 job (건드리면 안 됨)
- 한 에피소드 안에서 최대 3라운드: 순찰 → 제안 전부 승인 → 결과 대기 → (안 풀렸으면) 다음 순찰
  → "몇 라운드 만에 해결했나"를 잰다. 같은 설정으로 재시작했다가 같은 shard에서 또 멈추면 1라운드를 버린 것

지표 (조건별, 에피소드별)
- 해결률(3라운드 안에), 해결까지 라운드 수, 에피소드당 Agent 비용, 정상 job 건드림

실행 (프로젝트 루트에서):
  python eval/run_memory_eval.py --episodes 3     # 조건 2개 × 3 에피소드 (약 30분, Agent 비용 약 $0.3)
"""

import argparse
import asyncio
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "eval")
sys.path.insert(0, os.path.join(ROOT, "ops"))
sys.path.insert(0, os.path.join(ROOT, "agent"))

MAX_ROUNDS = 3
WAIT_START_S = 25
ROUND_TIMEOUT_S = 150
PRICE = {"gpt-5-mini": {"input": 0.25, "output": 2.00}}


def episode_jobs(rng):
    shard = rng.randint(1, 5)
    big = rng.choice([524288, 786432])
    base = 256 if big == 524288 else 512
    return [
        {"name": "stall_shard", "target": True, "detail": f"shard {shard}",
         "args": ["--step-time", "0.03", "--ckpt-every", "40", "--steps", "400", "--shard-steps", "30",
                  "--stall-shards", str(shard)]},
        {"name": "oom", "target": True, "detail": f"stage2_batch {big}",
         "args": ["--step-time", "0.03", "--ckpt-every", "40", "--steps", "110", "--batch-size", str(base),
                  "--stage2-at", "90", "--stage2-batch", str(big)]},
        {"name": "healthy", "target": False, "detail": "",
         "args": ["--step-time", "0.05", "--ckpt-every", "50", "--steps", "20000"]},
    ]


def launch(runs_dir, gpu, specs, rng):
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    truth = {}
    for spec in rng.sample(specs, len(specs)):
        job_id = f"run-{rng.randrange(16 ** 4):04x}"
        truth[job_id] = spec
        cmd = [sys.executable, os.path.join(ROOT, "sim", "fake_train.py"), "--job-id", job_id, "--gpu", str(gpu),
               "--runs-dir", runs_dir] + spec["args"]
        subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return truth


def finished_ok(jobs_mod, job_id):
    lines = jobs_mod.tail(job_id, 100000)
    starts = [i for i, l in enumerate(lines) if "Start training" in l]
    seg = lines[starts[-1]:] if starts else lines
    if not any("Exit 0" in l for l in seg):
        return False
    for l in reversed(seg):
        if (m := jobs_mod.LOSS_RE.search(l)):
            try:
                return math.isfinite(float(m.group(1)))
            except ValueError:
                return False
    return True


def settled(jobs_mod, job_id):
    """더 기다려도 바뀌지 않는 상태인가: 프로세스가 끝났거나, 살아 있는데 멈춰 있음"""
    pid = jobs_mod.read_job(job_id)["pid"]
    return (not jobs_mod._exists(pid)) or jobs_mod.is_stalled(job_id)


def kill_all(jobs_mod, truth):
    for j in truth:
        try:
            pid = jobs_mod.read_job(j)["pid"]
            if jobs_mod._exists(pid):
                os.kill(pid, signal.SIGKILL)
        except (ValueError, FileNotFoundError, ProcessLookupError):
            pass


async def run_episode(cond, ep, gpu, base_dir, rng):
    runs_dir = os.path.join(base_dir, f"{cond}_ep{ep}")
    os.environ["OPS_RUNS_DIR"] = runs_dir
    os.environ["OPS_STATE_DIR"] = os.path.join(runs_dir, "_state")      # 승인 대기열은 에피소드마다 새로
    os.environ["OPS_RUNBOOK_DIR"] = os.path.join(base_dir, f"{cond}_memory")  # runbook은 조건 안에서 계속 쌓임
    os.environ["OPS_MEMORY"] = "1" if cond == "memory_on" else "0"
    import agent
    import approvals
    import jobs as jobs_mod
    import runbooks
    jobs_mod.RUNS_DIR = runs_dir
    approvals.STATE_DIR = os.environ["OPS_STATE_DIR"]
    runbooks.STATE_DIR = os.environ["OPS_RUNBOOK_DIR"]

    specs = episode_jobs(rng)
    truth = launch(runs_dir, gpu, specs, rng)
    print(f"[{cond} ep{ep}] {[(s['name'], s['detail']) for s in specs]} → {WAIT_START_S}초 대기", file=sys.stderr)
    time.sleep(WAIT_START_S)

    targets = [j for j, t in truth.items() if t["target"]]
    resolved_at = {}
    cost = 0.0
    rounds_log = []
    try:
        for rnd in range(1, MAX_ROUNDS + 1):
            open_jobs = [j for j in targets if j not in resolved_at]
            if not open_jobs:
                break
            signal.signal(signal.SIGCHLD, signal.SIG_DFL)
            r = await agent.run_once()   # 안에서 verify_outcomes() → 지난 라운드 결과가 runbook에 기록됨
            t = r["trace"]
            price = PRICE.get(agent.MODEL, {"input": 0, "output": 0})
            cost += (t["input_tokens"] * price["input"] + t["output_tokens"] * price["output"]) / 1e6
            used_memory = any(c["name"] == "find_runbooks" for c in t["tool_calls"])
            approved = []
            for it in approvals.pending():
                approvals.decide(it["id"], approve=True, decided_by="eval-auto")
                approved.append({"job": truth[it["job_id"]]["name"], "action": it["action"],
                                 "changes": it.get("changes")})
            rounds_log.append({"round": rnd, "approved": approved, "find_runbooks_called": used_memory,
                               "runbooks_seen": sum(1 for c in t["tool_calls"] if c["name"] == "find_runbooks")})
            print(f"[{cond} ep{ep}] round {rnd}: {approved}", file=sys.stderr)
            time.sleep(8)  # 재시작된 프로세스가 새 pid를 쓸 시간
            deadline = time.time() + ROUND_TIMEOUT_S
            while time.time() < deadline and not all(settled(jobs_mod, j) for j in open_jobs):
                time.sleep(3)
            for j in open_jobs:
                if finished_ok(jobs_mod, j):
                    resolved_at[j] = rnd
        approvals.verify_outcomes()  # 마지막 라운드 결과도 runbook에 남긴다 (다음 에피소드가 쓴다)
    finally:
        kill_all(jobs_mod, truth)

    touched = any(it["job_id"] in truth and not truth[it["job_id"]]["target"] for it in approvals._load())
    return {
        "condition": cond, "episode": ep, "cost_usd": cost, "touched_healthy": touched, "rounds": rounds_log,
        "jobs": {truth[j]["name"]: {"detail": truth[j]["detail"], "resolved_round": resolved_at.get(j)} for j in targets},
        "runbooks_after": len(runbooks._load()) if cond == "memory_on" else 0,
    }


def summarize(results):
    print()
    for cond in ("memory_off", "memory_on"):
        rs = [r for r in results if r["condition"] == cond]
        print(f"=== {cond} ===")
        print(f"{'ep':<4}{'shard 멈춤':>14}{'OOM':>10}{'비용($)':>10}{'정상 job':>10}   runbook 수")
        for r in rs:
            fmt = lambda x: f"R{x}" if x else "실패"  # noqa: E731
            print(f"{r['episode']:<4}{fmt(r['jobs']['stall_shard']['resolved_round']):>14}"
                  f"{fmt(r['jobs']['oom']['resolved_round']):>10}{r['cost_usd']:>10.4f}"
                  f"{'건드림' if r['touched_healthy'] else '-':>10}   {r['runbooks_after']}")
        n = sum(len(r["jobs"]) for r in rs)
        solved = [x["resolved_round"] for r in rs for x in r["jobs"].values() if x["resolved_round"]]
        print(f"해결 {len(solved)}/{n}, 평균 라운드 {sum(solved) / len(solved) if solved else float('nan'):.2f}, "
              f"에피소드당 비용 ${sum(r['cost_usd'] for r in rs) / len(rs):.4f}\n")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    base_dir = os.path.join(EVAL_DIR, "runs", "mem_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    results = []
    for cond in ("memory_off", "memory_on"):
        rng = random.Random(args.seed)  # 두 조건이 같은 장애 순서(같은 shard 번호, 같은 batch)를 보게
        for ep in range(1, args.episodes + 1):
            results.append(await run_episode(cond, ep, args.gpu, base_dir, rng))
    summarize(results)

    out_dir = os.path.join(EVAL_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + "_memory.json")
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"자세한 결과: {out}")


if __name__ == "__main__":
    asyncio.run(main())
