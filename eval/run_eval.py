"""
평가: 실패를 주입한 job들을 띄우고 → Agent와 규칙 베이스라인이 같은 상황을 보게 한 뒤 → 정답과 비교

week1 평가와 같은 구조 (케이스 × 방법 → 지표). 다른 점
- 케이스가 "질문"이 아니라 "실제로 돌아가는 GPU job"이다
- 정답(--fail)은 이 스크립트만 알고, job 폴더에는 남기지 않는다
- job_id는 랜덤 이름으로 붙인다. "hang1" 같은 이름이면 Agent가 이름만 보고 맞힐 수 있다 (정답 누출)
- Agent의 "말"(보고서)뿐 아니라 "행동"(승인 대기열에 실제로 들어간 요청)을 채점한다

지표
- 진단 정확도       : diagnosis가 정답과 같은가 (Agent만. 규칙은 진단을 못 한다)
- false kill        : 종료하면 안 되는 job에 terminate를 요청 ← 가장 비싼 실수
- 놓침(miss)        : 종료해야 하는 job에 terminate를 요청하지 않음
- 비용, 시간

실행 (프로젝트 루트에서):
  python eval/run_eval.py              # 시나리오 1회 (약 2~3분, Agent 비용 약 $0.03)
  python eval/run_eval.py --repeat 3
"""

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "eval")

# 정답표: 주입한 실패 → 기대하는 진단과 조치
# (steps가 짧은 none은 평가 시점에 이미 끝나 있으므로 finished)
SCENARIO = [
    {"fail": "none",             "steps": 3000, "expect": "healthy",              "action": "none"},
    {"fail": "none",             "steps": 100,  "expect": "finished",             "action": "none"},
    {"fail": "slow",             "steps": 3000, "expect": "slow_but_progressing", "action": "none"},
    {"fail": "hang",             "steps": 3000, "expect": "hung",                 "action": "terminate"},
    {"fail": "hang_at_shutdown", "steps": 100,  "expect": "hung_at_shutdown",     "action": "terminate"},
    {"fail": "crash",            "steps": 3000, "expect": "crashed",              "action": "none"},
    {"fail": "oom",              "steps": 3000, "expect": "oom",                  "action": "none"},
    {"fail": "nan",              "steps": 3000, "expect": "nan_loss",             "action": "terminate"},
]
WAIT_S = 30                      # job을 띄우고 이만큼 기다린 뒤 평가 (hang_at_shutdown, finished가 끝나 있을 시간)
BASELINE_THRESHOLDS = [10, 15, 30, 60]   # 규칙 A: 로그가 N초 넘게 조용하면 HANG (vb_hang_killer 방식)
RELATIVE_K = [5, 10]                     # 규칙 B: 로그가 "그 job의 평소 간격 × K"보다 오래 조용하면 HANG
PRICE = {"gpt-5-mini": {"input": 0.25, "output": 2.00}}  # USD / 1M 토큰


def launch(runs_dir, gpu):
    """시나리오의 job들을 랜덤 id로 띄운다. 정답표(truth)는 여기서만 들고 있다."""
    # 이 스크립트가 job의 "부모 프로세스"라서, 끝난 job을 회수(wait)하지 않으면 좀비로 남는다.
    # 좀비는 os.kill(pid, 0)에 "살아 있다"고 답하므로 규칙 베이스라인이 끝난 job을 HANG으로 오판한다.
    # 실제 서버에서는 큐 스크립트/nohup이 회수해 주므로, 평가도 그 조건을 맞춘다 (SIG_IGN = 커널이 자동 회수)
    import signal
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    truth = {}
    jobs_ = SCENARIO[:]
    random.shuffle(jobs_)  # 띄우는 순서도 섞는다
    for spec in jobs_:
        job_id = f"run-{random.randrange(16 ** 4):04x}"
        truth[job_id] = spec
        cmd = [sys.executable, os.path.join(ROOT, "sim", "fake_train.py"), "--job-id", job_id,
               "--fail", spec["fail"], "--steps", str(spec["steps"]), "--fail-at", "50",
               "--gpu", str(gpu), "--runs-dir", runs_dir]
        subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return truth


def run_baseline(runs_dir, threshold):
    """네가 만든 rule_watchdog의 judge()를 그대로 쓴다. HANG이면 terminate로 간주."""
    sys.path.insert(0, os.path.join(ROOT, "baseline"))
    import rule_watchdog as rw
    rw.RUNS_DIR = runs_dir
    return {os.path.basename(d): ("terminate" if rw.judge(d, threshold) == "HANG" else "none")
            for d in rw.find_jobs()}


def run_relative_baseline(jobs_mod, k):
    """Agent에게 준 것과 같은 신호(평소 간격 대비 배수)를 규칙으로 쓰면 어떨까? → 공정한 비교를 위한 베이스라인"""
    acts = {}
    for j in jobs_mod.list_job_ids():
        s = jobs_mod.summary(j, gpu_mem={})
        ratio = s["log_age_vs_interval"]
        acts[j] = "terminate" if s["alive"] and ratio is not None and ratio > k else "none"
    return acts


def score_actions(truth, actions):
    false_kill = [j for j, t in truth.items() if t["action"] == "none" and actions.get(j) == "terminate"]
    missed = [j for j, t in truth.items() if t["action"] == "terminate" and actions.get(j) != "terminate"]
    return false_kill, missed


def cleanup(jobs_mod, truth):
    import signal
    for job_id in truth:
        try:
            pid = jobs_mod.read_job(job_id)["pid"]
            if jobs_mod.is_alive(pid, job_id):
                os.kill(pid, signal.SIGKILL)
        except (ValueError, FileNotFoundError, ProcessLookupError):
            pass


async def run_one(run_idx, gpu):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = os.path.join(EVAL_DIR, "runs", f"{stamp}_{run_idx}")
    state_dir = os.path.join(runs_dir, "_state")
    # Agent(그리고 Agent가 띄우는 MCP 서버)가 이 평가 전용 폴더만 보게 한다
    os.environ["OPS_RUNS_DIR"] = runs_dir
    os.environ["OPS_STATE_DIR"] = state_dir
    sys.path.insert(0, os.path.join(ROOT, "ops"))
    sys.path.insert(0, os.path.join(ROOT, "agent"))
    import approvals
    import jobs as jobs_mod
    jobs_mod.RUNS_DIR, approvals.STATE_DIR = runs_dir, state_dir
    import agent

    truth = launch(runs_dir, gpu)
    print(f"[run {run_idx}] job {len(truth)}개 실행, {WAIT_S}초 대기...", file=sys.stderr)
    time.sleep(WAIT_S)

    try:
        # 같은 순간의 상태로 베이스라인 판정 (Agent는 순찰에 1~2분 걸리므로 먼저)
        baseline = {f"규칙A (절대 {th}s)": run_baseline(runs_dir, th) for th in BASELINE_THRESHOLDS}
        baseline.update({f"규칙B (평소 간격×{k})": run_relative_baseline(jobs_mod, k) for k in RELATIVE_K})
        # 순찰 중에는 자동 회수를 끈다. MCP 클라이언트가 자기 서버 프로세스의 종료 코드를 받아야 하기 때문
        # (이 사이에 끝나는 job이 좀비가 되어도 ops/jobs.py의 is_alive는 좀비를 "죽음"으로 처리한다)
        import signal
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        print(f"[run {run_idx}] Agent 순찰 중...", file=sys.stderr)
        result = await agent.run_once()
        # 행동 채점: 보고서에 뭐라고 썼든, 승인 대기열에 실제로 들어간 요청이 기준
        requested = {it["job_id"]: it["action"] for it in approvals._load() if it["action"] == "terminate"}
    finally:
        cleanup(jobs_mod, truth)

    diag = {j["job_id"]: j["diagnosis"] for j in result["report"]["jobs"]}
    t = result["trace"]
    price = PRICE.get(agent.MODEL, {"input": 0, "output": 0})
    cost = (t["input_tokens"] * price["input"] + t["output_tokens"] * price["output"]) / 1e6
    fk, miss = score_actions(truth, requested)
    return {
        "run": run_idx,
        "truth": truth,
        "agent": {
            "diagnosis": diag,
            "requested": requested,
            "diag_correct": {j: diag.get(j) == truth[j]["expect"] for j in truth},
            "false_kill": fk, "missed": miss,
            "cost_usd": cost, "latency_s": t["latency_s"], "llm_calls": t["llm_calls"],
            "tool_calls": len(t["tool_calls"]),
            "report": result["report"],
        },
        "baseline": {th: dict(zip(["false_kill", "missed"], score_actions(truth, acts)))
                     for th, acts in baseline.items()},
    }


def summarize(runs):
    n_runs = len(runs)
    n_jobs = sum(len(r["truth"]) for r in runs)
    n_safe = sum(sum(t["action"] == "none" for t in r["truth"].values()) for r in runs)
    n_bad = n_jobs - n_safe

    print("\n=== 실패 유형별 Agent 진단 ===")
    by_type = {}
    for r in runs:
        for j, t in r["truth"].items():
            key = f"{t['fail']}({t['steps']})"
            by_type.setdefault(key, []).append((t["expect"], r["agent"]["diagnosis"].get(j, "(누락)"),
                                                r["agent"]["requested"].get(j, "none"), t["action"]))
    for key, rows in by_type.items():
        ok = sum(e == d for e, d, _, _ in rows)
        got = ", ".join(f"{d}{'+T' if a == 'terminate' else ''}" for _, d, a, _ in rows)
        print(f"  {key:<24} 기대 {rows[0][0]:<22} {ok}/{len(rows)}   실제: {got}")

    print("\n=== 방법 비교 (조치 기준) ===")
    print(f"{'방법':<22}{'false kill':>14}{'놓침':>10}{'진단정확도':>12}{'비용/순찰':>12}{'시간/순찰':>10}")
    a = [r["agent"] for r in runs]
    fk = sum(len(x["false_kill"]) for x in a)
    ms = sum(len(x["missed"]) for x in a)
    acc = sum(sum(x["diag_correct"].values()) for x in a)
    print(f"{'Agent':<22}{f'{fk}/{n_safe}':>14}{f'{ms}/{n_bad}':>10}{f'{acc}/{n_jobs}':>12}"
          f"{'$' + format(sum(x['cost_usd'] for x in a) / n_runs, '.4f'):>12}"
          f"{format(sum(x['latency_s'] for x in a) / n_runs, '.0f') + 's':>10}")
    for name in runs[0]["baseline"]:
        b = [r["baseline"][name] for r in runs]
        fk = sum(len(x["false_kill"]) for x in b)
        ms = sum(len(x["missed"]) for x in b)
        print(f"{name:<22}{f'{fk}/{n_safe}':>14}{f'{ms}/{n_bad}':>10}{'-':>12}{'$0':>12}{'~0s':>10}")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--gpu", type=int, default=1)
    args = p.parse_args()

    runs = [await run_one(i, args.gpu) for i in range(args.repeat)]
    summarize(runs)

    out_dir = os.path.join(EVAL_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    with open(out, "w") as f:
        json.dump(runs, f, ensure_ascii=False, indent=2)
    print(f"\n자세한 결과: {out}")


if __name__ == "__main__":
    asyncio.run(main())
