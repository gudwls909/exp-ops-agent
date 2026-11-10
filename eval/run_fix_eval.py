"""
수정(fix) 평가: Agent가 제안한 "설정을 고친 재시작"이 실제로 job을 살리는가, 그리고 실험을 바꾸지 않았는가

run_eval.py(감지 평가)와 다른 점
- 감지만 맞히면 끝이 아니다. 승인된 수정으로 다시 돌린 job이 "끝까지 정상 완료"됐는지를 결과로 채점한다
- 하지만 "끝까지 돌았다"만 보면 batch를 1000배 줄인 수정도 성공이 된다 (troubleshooting_log 10번)
  → "실험 의미 보존" 검사를 따로 한다 (정답 기준은 이 스크립트만 안다)

방법 비교
- Agent       : Agent가 보고서에 쓴 수정을 전부 승인 (사람이 검토 없이 다 승인하는 최악의 경우 = 제안 품질 그 자체)
- 그냥 재시작   : 설정을 안 바꾸고 마지막 체크포인트부터 재시작 (규칙 기반 자동 재시작이 할 수 있는 최선)

지표 (실패 job마다)
- 수정 성공     : 재시작한 job이 끝까지 돌고(Exit 0) 마지막 loss가 정상(유한하고 < 1)
- 의미 보존     : 바꾼 설정이 원인과 관련된 것뿐이고, 실험의 정체(실질 batch, 데이터 등)를 바꾸지 않았나
- 좋은 수정     : 수정 성공 AND 의미 보존
- 잘못 건드림   : 정상 job에 조치를 요청

실행 (프로젝트 루트에서):
  python eval/run_fix_eval.py              # 1회 = Agent 1번 + 그냥 재시작 1번 (약 8분, Agent 비용 약 $0.02)
  python eval/run_fix_eval.py --repeat 2
  python eval/run_fix_eval.py --set heldout --repeat 2   # 개발 중에 보지 않은 조건
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

COMMON = ["--step-time", "0.05", "--ckpt-every", "50"]
WAIT_S = 25          # 띄우고 순찰까지
FINISH_TIMEOUT_S = 300
PRICE = {"gpt-5-mini": {"input": 0.25, "output": 2.00}}


# ---------------------------------------------------------------
# 시나리오와 "좋은 수정"의 기준 (정답). Agent는 이 기준을 모른다
# ---------------------------------------------------------------
def ok_diverge(ch, resume):
    # 원인: stage2_lr=1e3. 좋은 수정 = 학습률 관련 설정만 낮추고, 발산 전(step 150 이전) 체크포인트부터
    lr_keys = {"stage2_lr", "max_lr", "warmup_steps"}
    lr_ok = ch.get("stage2_lr") is not None and ch["stage2_lr"] <= 1e-2
    return set(ch) <= lr_keys and lr_ok and resume is not None and resume < 150


def ok_oom(ch, resume):
    # 원인: stage2_batch가 너무 큼. 좋은 수정 = 실질 batch(stage2_batch)는 그대로, grad_accum만 늘림
    return set(ch) == {"grad_accum"} and ch["grad_accum"] >= 2


def ok_shard(ch, resume):
    # 원인: shard 3 손상. 좋은 수정 = 3번만 빼고 나머지는 그대로
    return set(ch) == {"data_shards"} and ch["data_shards"] == "0,1,2,4,5,6,7"


def ok_hang(ch, resume):
    # 원인: 일시적 멈춤. 좋은 수정 = 설정 변경 없이 재시작
    return ch == {}


# ---- held-out용 기준 (숫자가 다르다) ----
def ok_diverge_h(ch, resume):
    return set(ch) <= {"stage2_lr", "max_lr", "warmup_steps"} and (ch.get("stage2_lr") or 1) <= 1e-2 \
        and resume is not None and resume < 250


def ok_shard_h(ch, resume):
    return set(ch) == {"data_shards"} and ch["data_shards"] == "0,1,2,3,4"


def ok_oom_h(ch, resume):
    # grad_accum 2로는 여전히 OOM이 난다(직접 측정). 3 이상이어야 실제로 돈다 → "수정 성공"에서 걸러짐
    return set(ch) == {"grad_accum"} and ch["grad_accum"] >= 2


# kind: restart(고쳐서 재시작해야 함) / terminate(종료만 해야 함) / safe(건드리면 안 됨)
SETS = {
    "dev": [  # 개발하면서 보고 4번 고친 시나리오
        {"name": "diverge", "kind": "restart", "check": ok_diverge,
         "args": COMMON + ["--steps", "1500", "--stage2-at", "150", "--stage2-lr", "1e3"]},
        {"name": "oom", "kind": "restart", "check": ok_oom,
         "args": COMMON + ["--steps", "150", "--stage2-at", "120", "--stage2-batch", "524288"]},
        {"name": "bad_shard", "kind": "restart", "check": ok_shard,
         "args": COMMON + ["--steps", "300", "--bad-shards", "3"]},
        {"name": "hang", "kind": "restart", "check": ok_hang,
         "args": COMMON + ["--steps", "300", "--fail", "hang", "--fail-at", "70"]},
        {"name": "healthy", "kind": "safe", "check": None, "args": COMMON + ["--steps", "8000"]},
    ],
    # held-out: 한 번도 돌려 보지 않은 조건. ⚠️ 이 결과를 보고 고치면 안 된다
    #   다른 점: 숫자(lr, batch, 시점), 학습 속도, 체크포인트 간격, shard 구성, grad_accum 2로는 부족한 OOM,
    #   그리고 새 유형 2개: "종료만 해야 하는 job"(hang_at_shutdown), "건드리면 안 되는 느린 job"
    "heldout": [
        {"name": "diverge_h", "kind": "restart", "check": ok_diverge_h,
         "args": ["--step-time", "0.02", "--ckpt-every", "40", "--steps", "3000", "--stage2-at", "250",
                  "--stage2-lr", "5e2"]},
        {"name": "oom_h", "kind": "restart", "check": ok_oom_h,
         "args": ["--step-time", "0.03", "--ckpt-every", "40", "--steps", "110", "--batch-size", "512",
                  "--stage2-at", "90", "--stage2-batch", "786432"]},
        {"name": "shard_h", "kind": "restart", "check": ok_shard_h,
         "args": ["--step-time", "0.03", "--ckpt-every", "40", "--steps", "400", "--data-shards", "0,1,2,3,4,5",
                  "--shard-steps", "30", "--bad-shards", "5"]},
        {"name": "hang_h", "kind": "restart", "check": ok_hang,
         "args": ["--step-time", "0.2", "--log-every", "5", "--ckpt-every", "40", "--steps", "250",
                  "--fail", "hang", "--fail-at", "60"]},
        {"name": "shutdown_h", "kind": "terminate", "check": None,
         "args": ["--step-time", "0.1", "--ckpt-every", "40", "--steps", "80", "--fail", "hang_at_shutdown"]},
        {"name": "slow_h", "kind": "safe", "check": None,
         "args": ["--step-time", "0.1", "--ckpt-every", "50", "--steps", "5000", "--fail", "slow",
                  "--fail-at", "30", "--slow-factor", "10"]},
        {"name": "healthy_h", "kind": "safe", "check": None,
         "args": ["--step-time", "0.05", "--ckpt-every", "50", "--steps", "8000"]},
    ],
}


def launch(runs_dir, gpu, scenario):
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)  # 끝난 job 자동 회수 (run_eval.py와 같은 이유)
    truth = {}
    for spec in random.sample(scenario, len(scenario)):
        job_id = f"run-{random.randrange(16 ** 4):04x}"
        truth[job_id] = spec
        cmd = [sys.executable, os.path.join(ROOT, "sim", "fake_train.py"), "--job-id", job_id, "--gpu", str(gpu),
               "--runs-dir", runs_dir] + spec["args"]
        subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return truth


def outcome(jobs_mod, job_id):
    """재시작 후 결과: 끝까지 정상 완료했나. (재시작 이후의 로그만 본다)"""
    lines = jobs_mod.tail(job_id, 100000)
    starts = [i for i, l in enumerate(lines) if "Start training" in l]
    after = lines[starts[-1]:] if starts else lines
    done = any("Exit 0" in l for l in after)
    loss = None
    for l in reversed(after):
        if (m := jobs_mod.LOSS_RE.search(l)):
            loss = m.group(1)
            break
    try:
        finite = loss is not None and math.isfinite(float(loss)) and float(loss) < 1.0
    except ValueError:
        finite = False
    return {"completed": done and finite, "exit0": done, "last_loss": loss}


def wait_finished(jobs_mod, job_ids):
    deadline = time.time() + FINISH_TIMEOUT_S
    while time.time() < deadline:
        alive = [j for j in job_ids if jobs_mod._exists(jobs_mod.read_job(j)["pid"])]
        if not alive:
            return
        time.sleep(3)


def kill_all(jobs_mod, truth):
    for j in truth:
        try:
            pid = jobs_mod.read_job(j)["pid"]
            if jobs_mod._exists(pid):
                os.kill(pid, signal.SIGKILL)
        except (ValueError, FileNotFoundError, ProcessLookupError):
            pass


async def run_one(idx, gpu, method, set_name):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = os.path.join(EVAL_DIR, "runs", f"fix_{set_name}_{stamp}_{idx}_{method}")
    os.environ["OPS_RUNS_DIR"] = runs_dir
    os.environ["OPS_STATE_DIR"] = os.path.join(runs_dir, "_state")
    import agent
    import approvals
    import jobs as jobs_mod
    jobs_mod.RUNS_DIR, approvals.STATE_DIR = runs_dir, os.environ["OPS_STATE_DIR"]

    truth = launch(runs_dir, gpu, SETS[set_name])
    print(f"[{idx}/{method}] job {len(truth)}개 실행, {WAIT_S}초 대기...", file=sys.stderr)
    time.sleep(WAIT_S)
    result = {"method": method, "truth": {j: t["name"] for j, t in truth.items()}, "jobs": {}}
    try:
        if method == "agent":
            signal.signal(signal.SIGCHLD, signal.SIG_DFL)  # MCP 클라이언트가 서버 종료 코드를 받을 수 있게
            print(f"[{idx}/{method}] Agent 순찰 중...", file=sys.stderr)
            r = await agent.run_once()
            t = r["trace"]
            price = PRICE.get(agent.MODEL, {"input": 0, "output": 0})
            result["cost_usd"] = (t["input_tokens"] * price["input"] + t["output_tokens"] * price["output"]) / 1e6
            result["latency_s"] = t["latency_s"]
            result["report"] = r["report"]
            proposals = {it["job_id"]: it for it in approvals._load()}
        else:
            # 그냥 재시작: 실패 job(정상 job 제외)을 설정 변경 없이 마지막 체크포인트부터
            #   (검증기가 "망가진 체크포인트"라며 거부하면 재시작을 못 한 것으로 남는다)
            proposals = {}
            for j, t in truth.items():
                if t["kind"] != "safe":
                    try:
                        proposals[j] = approvals.request(j, "restart", "plain restart")
                    except ValueError:
                        pass

        # 전부 승인 (검토 없이) → 실제로 실행
        relaunched = []
        for j, it in proposals.items():
            if it["status"] == "pending":
                approvals.decide(it["id"], approve=True, decided_by="eval-auto")
                relaunched.append(j)
        print(f"[{idx}/{method}] 승인·재시작 {len(relaunched)}개, 완료 대기...", file=sys.stderr)
        time.sleep(8)  # 재시작된 프로세스가 job.json에 새 pid를 쓸 시간 (안 기다리면 옛 pid를 보고 "끝났다"고 착각)
        wait_finished(jobs_mod, [j for j in relaunched if truth[j]["kind"] != "safe"])

        for j, t in truth.items():
            p = proposals.get(j)
            row = {"scenario": t["name"], "action": p["action"] if p else "none",
                   "changes": p.get("changes") if p else None, "resume_from": p.get("resume_from") if p else None}
            if t["kind"] == "safe":
                row["touched"] = p is not None
            elif t["kind"] == "terminate":
                # 학습은 이미 끝났다 → 종료만 하면 됨. 재시작하면 같은 자리에서 또 멈춘다
                row["outcome"] = {"completed": bool(p) and p["action"] == "terminate"}
                row["semantic_ok"] = row["good_fix"] = row["outcome"]["completed"]
            else:
                row["outcome"] = outcome(jobs_mod, j) if p and p["action"] == "restart" else {"completed": False}
                row["semantic_ok"] = bool(p) and p["action"] == "restart" and t["check"](p.get("changes") or {},
                                                                                         p.get("resume_from"))
                row["good_fix"] = row["outcome"]["completed"] and row["semantic_ok"]
            result["jobs"][j] = row
    finally:
        kill_all(jobs_mod, truth)
    return result


def summarize(results, set_name):
    names = [s["name"] for s in SETS[set_name] if s["kind"] != "safe"]
    safe_names = {s["name"] for s in SETS[set_name] if s["kind"] == "safe"}
    for method in ("agent", "plain_restart"):
        rs = [r for r in results if r["method"] == method]
        if not rs:
            continue
        print(f"\n=== {method} ({len(rs)}회) ===")
        print(f"{'시나리오':<12}{'수정 성공':>10}{'의미 보존':>10}{'좋은 수정':>10}   제안")
        tot = {"c": 0, "s": 0, "g": 0, "n": 0}
        for name in names:
            rows = [row for r in rs for row in r["jobs"].values() if row["scenario"] == name]
            c = sum(row["outcome"]["completed"] for row in rows)
            s = sum(row["semantic_ok"] for row in rows)
            g = sum(row["good_fix"] for row in rows)
            tot["c"] += c; tot["s"] += s; tot["g"] += g; tot["n"] += len(rows)
            props = "; ".join(f"{row['changes']}@{row['resume_from'] or 'last'}" if row["action"] == "restart"
                              else row["action"] for row in rows)
            print(f"{name:<12}{f'{c}/{len(rows)}':>10}{f'{s}/{len(rows)}':>10}{f'{g}/{len(rows)}':>10}   {props}")
        touched = sum(row.get("touched", False) for r in rs for row in r["jobs"].values())
        c, s_, g = (f"{tot[k]}/{tot['n']}" for k in ("c", "s", "g"))
        n_safe = sum(row["scenario"] in safe_names for r in rs for row in r["jobs"].values())
        print(f"{'합계':<12}{c:>10}{s_:>10}{g:>10}   정상 job 건드림: {touched}/{n_safe}")
        if method == "agent":
            print(f"비용/순찰 ${sum(r['cost_usd'] for r in rs) / len(rs):.4f}, 시간/순찰 "
                  f"{sum(r['latency_s'] for r in rs) / len(rs):.0f}s")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--set", choices=list(SETS), default="dev")
    args = p.parse_args()

    results = []
    for i in range(args.repeat):
        results.append(await run_one(i, args.gpu, "agent", args.set))
        if not args.skip_baseline:
            results.append(await run_one(i, args.gpu, "plain_restart", args.set))
    summarize(results, args.set)

    out_dir = os.path.join(EVAL_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + f"_fix_{args.set}.json")
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n자세한 결과: {out}")


if __name__ == "__main__":
    asyncio.run(main())
