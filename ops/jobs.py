"""
job 조회/조작 라이브러리 — MCP 서버, 승인 CLI, 테스트가 모두 이 파일의 함수를 쓴다

baseline/rule_watchdog.py에서 직접 짠 함수들(is_alive, log_age)을 "실전용"으로 보강한 버전이다.
보강한 것
- job_id 검증         : week1 read_file의 경로 탐색 방어와 같은 이유. "../../etc" 같은 job_id를 막는다
- PermissionError     : 다른 사용자의 프로세스는 os.kill(pid, 0)에서 PermissionError → "존재함"으로 본다
- pid 재사용 / 좀비    : job이 죽은 뒤 같은 pid를 다른 프로세스가 받을 수 있다.
                         /proc/<pid>/cmdline에 job_id가 있는지 확인해서, 엉뚱한 프로세스를 우리 job으로 착각(→ 잘못 kill)하지 않는다
- terminate           : SIGTERM 먼저 (체크포인트 저장 기회) → 안 죽으면 SIGKILL

⚠️ Agent(LLM)에게 job.json의 cmd를 그대로 보여 주지 않는다. --fail 옵션(정답)이 들어 있기 때문.
"""

import json
import os
import re
import signal
import statistics
import subprocess
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 테스트에서는 환경변수로 임시 폴더를 가리키게 한다 (week1 db.py의 DB_PATH와 같은 방식)
RUNS_DIR = os.getenv("OPS_RUNS_DIR", os.path.join(ROOT, "runs"))

JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
STEP_RE = re.compile(r"step (\d+)/(\d+)")
# 로그 한 줄 예: "... INFO step 230/300 loss 0.01574 lr 1.0e-03 gpu_mem 97MiB step_time 0.10s"
LOSS_RE = re.compile(r"loss (\S+)")                    # \S+ : 공백이 아닌 글자 1개 이상 → "0.01574", "nan", "inf" 모두
STEP_TIME_RE = re.compile(r"step_time (\d+\.?\d*)s")   # 숫자, (점 있을 수도), 숫자 0개 이상, 그리고 글자 s


# ---------------------------------------------------------------
# 기본 조회
# ---------------------------------------------------------------
def job_dir(job_id: str) -> str:
    """job_id를 검증하고 job 폴더 경로를 돌려준다. 이상한 id면 ValueError."""
    if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id) or ".." in job_id:
        raise ValueError(f"잘못된 job_id: {job_id!r}")
    d = os.path.join(RUNS_DIR, job_id)
    if not os.path.isfile(os.path.join(d, "job.json")):
        raise ValueError(f"존재하지 않는 job: {job_id}")
    return d


def list_job_ids() -> list[str]:
    if not os.path.isdir(RUNS_DIR):
        return []
    return sorted(n for n in os.listdir(RUNS_DIR)
                  if JOB_ID_RE.match(n) and os.path.isfile(os.path.join(RUNS_DIR, n, "job.json")))


def read_job(job_id: str) -> dict:
    with open(os.path.join(job_dir(job_id), "job.json")) as f:
        return json.load(f)


def _exists(pid: int) -> bool:
    """프로세스가 아직 완전히 끝나지 않았으면 True (좀비 = 끝난 것으로 본다)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/status") as f:
            return not any(line.startswith("State:") and "Z" in line.split()[1] for line in f)
    except FileNotFoundError:
        return False


def is_alive(pid: int, job_id: str) -> bool:
    """그 pid가 살아 있고, 그리고 정말 이 job의 프로세스인지까지 확인한다."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # 존재는 한다 (다른 사용자 프로세스). 아래 cmdline 검사로 넘어간다
    try:
        with open(f"/proc/{pid}/status") as f:
            if any(line.startswith("State:") and "Z" in line.split()[1] for line in f):
                return False  # 좀비: 이미 끝났는데 부모가 회수를 안 한 상태
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().replace(b"\0", b" ").decode(errors="replace")
    except FileNotFoundError:
        return False
    return job_id in cmdline.split()  # pid 재사용 방어: 명령줄에 이 job_id가 있어야 우리 job


def log_age(job_id: str) -> float:
    path = os.path.join(job_dir(job_id), "train.log")
    if not os.path.exists(path):
        return float("inf")
    return time.time() - os.path.getmtime(path)


def tail(job_id: str, n: int = 30) -> list[str]:
    path = os.path.join(job_dir(job_id), "train.log")
    if not os.path.exists(path):
        return []
    with open(path, errors="replace") as f:
        return f.read().splitlines()[-n:]


def last_step(job_id: str):
    """로그에서 마지막 'step X/Y'를 찾아 (X, Y)로. 없으면 (None, None)."""
    for line in reversed(tail(job_id, 500)):
        m = STEP_RE.search(line)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def last_metrics(job_id: str):
    """로그에서 마지막 loss(문자열)와 step_time(초)을 찾는다. 없으면 None.
    loss를 숫자로 바꾸지 않는 이유: float("nan")은 JSON 표준에 없어서 json.dumps가 NaN이라는 비표준 글자를 쓴다.
    로그에 찍힌 글자 그대로("nan") 넘기는 게 LLM에게도 파서에게도 안전하다."""
    loss = step_time = None
    for line in reversed(tail(job_id, 500)):
        if loss is None and (m := LOSS_RE.search(line)):
            loss = m.group(1)
        if step_time is None and (m := STEP_TIME_RE.search(line)):
            step_time = float(m.group(1))
        if loss is not None and step_time is not None:
            break
    return loss, step_time


def recent_log_interval(job_id: str, k: int = 6):
    """최근 step 로그 줄들 사이의 '평소 간격'(초, 중앙값). 계산할 수 없으면 None.
    log_age만으로는 "26초가 긴가?"를 알 수 없다. 1초마다 쓰던 job이면 26배 지연(멈춤)이고,
    20초마다 쓰던 job이면 정상이다. 그 job 자신의 속도와 비교해야 한다."""
    times = []
    for line in reversed(tail(job_id, 500)):
        if STEP_RE.search(line):
            try:
                times.append(datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S").timestamp())
            except ValueError:
                continue
            if len(times) > k:
                break
    gaps = [a - b for a, b in zip(times, times[1:])]
    if not gaps:
        return None
    return max(statistics.median(gaps), 1.0)  # 로그 시각이 초 단위라 1초보다 작게는 못 잰다


def read_config(job_id: str) -> dict:
    """학습 설정(config.json). 실패 주입 옵션은 들어 있지 않다 (시뮬레이터가 애초에 안 쓴다)."""
    path = os.path.join(job_dir(job_id), "config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


CKPT_RE = re.compile(r"^ckpt-(\d+)\.pt$")


def checkpoints(job_id: str) -> list[dict]:
    """step별 체크포인트 목록 + 그 step에 로그로 찍힌 loss. 어느 체크포인트가 "망가지기 전"인지 고를 때 쓴다."""
    # ⚠️ 체크포인트 N = "step N의 업데이트가 끝난 뒤"의 가중치. 그런데 step N 로그의 loss는 업데이트 "전"에 계산된 값이다.
    #    그래서 step N에서 lr이 폭주하면 loss_logged(N)은 정상인데 체크포인트 N은 이미 망가져 있다 (실제로 겪은 함정).
    #    이 체크포인트가 멀쩡한지는 "다음 로그의 loss"(이 가중치에서 출발한 결과)로 봐야 한다.
    d = job_dir(job_id)
    steps = sorted(int(m.group(1)) for n in os.listdir(d) if (m := CKPT_RE.match(n)))
    logged = []  # (step, loss) 순서대로. 재시작 등으로 같은 step이 또 나오면 뒤의 것이 이긴다
    for line in tail(job_id, 100000):
        m = STEP_RE.search(line)
        if m and (lm := LOSS_RE.search(line)):
            logged.append((int(m.group(1)), lm.group(1)))
    loss_at = dict(logged)
    out = []
    for st in steps:
        nxt = next((l for s_, l in sorted(loss_at.items()) if s_ > st), None)
        out.append({"step": st, "loss_logged_at_step": loss_at.get(st), "loss_at_next_log": nxt})
    return out


# ---------------------------------------------------------------
# GPU
# ---------------------------------------------------------------
def _nvidia_smi(query: str, kind: str = "gpu") -> list[list[str]]:
    flag = "--query-gpu" if kind == "gpu" else "--query-compute-apps"
    try:
        out = subprocess.run(["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def gpus() -> list[dict]:
    rows = _nvidia_smi("index,utilization.gpu,memory.used,memory.total")
    return [{"gpu": int(r[0]), "util_pct": int(r[1]), "mem_used_mib": int(r[2]), "mem_total_mib": int(r[3])}
            for r in rows]


def gpu_mem_by_pid() -> dict[int, int]:
    return {int(r[0]): int(r[1]) for r in _nvidia_smi("pid,used_memory", kind="apps") if r[0].isdigit()}


# ---------------------------------------------------------------
# 요약: Agent에게 보여 줄 정보 (cmd는 빼고!)
# ---------------------------------------------------------------
def summary(job_id: str, gpu_mem: dict | None = None) -> dict:
    job = read_job(job_id)
    pid = job["pid"]
    step, total = last_step(job_id)
    loss, step_time = last_metrics(job_id)
    interval = recent_log_interval(job_id)
    gpu_mem = gpu_mem_by_pid() if gpu_mem is None else gpu_mem
    age = log_age(job_id)
    return {
        "job_id": job_id,
        "alive": is_alive(pid, job_id),
        "pid": pid,
        "started_at": job.get("started_at"),
        "log_age_s": None if age == float("inf") else round(age, 1),
        "last_step": step,
        "total_steps": total,
        "last_loss": loss,
        "last_step_time_s": step_time,
        "recent_log_interval_s": interval,
        # 로그가 평소 간격의 몇 배만큼 조용한가. 1~2면 정상 범위, 수십이면 멈췄을 가능성이 크다
        "log_age_vs_interval": None if (interval is None or age == float("inf")) else round(age / interval, 1),
        "gpu_mem_mib": gpu_mem.get(pid),
    }


# ---------------------------------------------------------------
# 위험한 조작 — MCP 도구로 노출하지 않는다. 사람이 승인한 뒤 approvals.py만 부른다
# ---------------------------------------------------------------
def terminate(job_id: str, grace_s: float = 15.0) -> str:
    """SIGTERM → grace_s초 기다림 → 그래도 살아 있으면 SIGKILL."""
    pid = read_job(job_id)["pid"]
    if not is_alive(pid, job_id):
        return f"{job_id}: 이미 종료됨 (pid {pid}). 아무것도 하지 않음"
    os.kill(pid, signal.SIGTERM)
    # 기다릴 때는 is_alive가 아니라 _exists로 본다.
    # 종료 중인 프로세스는 /proc/<pid>/cmdline이 먼저 비어서 is_alive가 "죽었다"고 일찍 말할 수 있다
    # (체크포인트를 저장하는 중인데 "끝났다"고 보고하면, 바로 재시작했을 때 체크포인트가 깨질 수 있다)
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if not _exists(pid):
            return f"{job_id}: SIGTERM으로 정상 종료 (pid {pid})"
        time.sleep(0.2)
    os.kill(pid, signal.SIGKILL)
    while _exists(pid) and time.time() < deadline + 5:
        time.sleep(0.1)
    return f"{job_id}: SIGTERM에 반응 없음 → SIGKILL (pid {pid})"


# Agent가 바꾸자고 제안할 수 있는 설정과 허용 범위 (이 밖의 것은 전부 거부)
#   - 실험의 "정체"를 바꾸는 설정(steps, 모델 구조, seed 등)은 목록에 없다
#   - batch_size / stage2_batch도 뺐다: 프롬프트로 "batch는 줄이지 마라"고 해도 LLM이 3번 중 2번꼴로 줄였다
#     (troubleshooting_log 10, 15번). 실험 의미 보존은 프롬프트가 아니라 여기서 강제한다.
#     메모리가 모자라면 grad_accum으로만 해결할 수 있다 (실질 batch는 그대로)
#   - data_shards는 "빼기"만 허용한다 (손상된 shard 제외). 새 데이터를 넣는 건 사람이 할 일
CHANGEABLE = {
    "max_lr":       (float, 1e-6, 1.0),
    "stage2_lr":    (float, 1e-6, 1.0),
    "warmup_steps": (int, 1, 100_000),
    "grad_accum":   (int, 1, 1 << 20),  # 상한은 아래에서 "batch보다 클 수 없음"으로 다시 검사
    "data_shards":  (str, None, None),
}
BROKEN_RATIO = 100  # 체크포인트 다음 loss가 그 전 최저 loss의 100배를 넘으면 "이미 망가진 체크포인트"


def validate_changes(job_id: str, changes: dict | None, resume_from: int | None) -> tuple[dict, int | None]:
    """제안된 변경을 검사해서 정리된 형태로 돌려준다. 하나라도 이상하면 ValueError (승인 요청 자체를 만들지 않는다)."""
    changes = {k: v for k, v in (changes or {}).items() if v is not None}
    config = read_config(job_id)
    clean = {}
    for key, value in changes.items():
        if key not in CHANGEABLE:
            raise ValueError(f"바꿀 수 없는 설정: {key} (가능: {sorted(CHANGEABLE)})")
        typ, lo, hi = CHANGEABLE[key]
        if key == "data_shards":
            new = [int(x) for x in str(value).split(",") if x.strip() != ""]
            old = [int(x) for x in str(config.get("data_shards", "")).split(",") if x.strip() != ""]
            if not new or not set(new) <= set(old):
                raise ValueError(f"data_shards는 기존 shard {old}에서 빼는 것만 가능: {new}")
            # 데이터를 빼는 건 실험을 바꾸는 일 → 이 job의 로그에 "그 shard가 손상됐다"는 증거가 있을 때만 허용
            log = "\n".join(tail(job_id, 100000))
            no_proof = [k for k in set(old) - set(new) if f"corrupted shard data/shard-{k:04d}" not in log]
            if no_proof:
                raise ValueError(f"shard {sorted(no_proof)}가 손상됐다는 증거가 이 job 로그에 없다 (손상이 확인된 shard만 뺄 수 있다)")
            clean[key] = ",".join(map(str, new))
            continue
        try:
            v = typ(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} 값이 이상함: {value!r}")
        if not (lo <= v <= hi):
            raise ValueError(f"{key}={v}는 허용 범위 [{lo}, {hi}] 밖")
        if key == "grad_accum":
            biggest = max(int(config.get("batch_size", 1)), int(config.get("stage2_batch", 1)))
            if v > biggest:
                raise ValueError(f"grad_accum={v}는 batch({biggest})보다 클 수 없음")
        clean[key] = v
    ckpts = checkpoints(job_id)
    steps = [c["step"] for c in ckpts]
    if resume_from is not None:
        if int(resume_from) not in steps:
            raise ValueError(f"step {resume_from} 체크포인트 없음 (있는 것: {steps})")
        resume_from = int(resume_from)
    # 망가진 체크포인트에서 재시작하면 설정을 고쳐도 소용없다 (troubleshooting_log 13번)
    target = resume_from if resume_from is not None else (steps[-1] if steps else None)
    if target is not None:
        _check_not_broken(ckpts, target)
    return clean, resume_from


def _loss_value(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _check_not_broken(ckpts: list[dict], step: int) -> None:
    import math
    healthy = [v for c in ckpts if c["step"] < step and (v := _loss_value(c["loss_logged_at_step"])) is not None
               and math.isfinite(v)]
    nxt_text = next(c["loss_at_next_log"] for c in ckpts if c["step"] == step)
    nxt = _loss_value(nxt_text)
    if nxt is None:
        if nxt_text is not None:  # "nan" 같은 글자
            raise ValueError(f"step {step} 체크포인트 다음 loss가 {nxt_text} → 이미 망가진 체크포인트. 더 이른 step을 지정하라")
        return  # 다음 로그가 아직 없음 (방금 저장된 체크포인트) → 판단 불가, 통과
    if not math.isfinite(nxt) or (healthy and nxt > BROKEN_RATIO * min(healthy)):
        raise ValueError(f"step {step} 체크포인트 다음 loss가 {nxt_text} (그 전 최저 {min(healthy) if healthy else '?'}) "
                         f"→ 이미 망가진 체크포인트. 더 이른 step을 지정하라 (있는 것: {[c['step'] for c in ckpts]})")


def _without_flags(cmd: list[str], flags: set[str]) -> list[str]:
    """명령줄에서 '--flag 값' (또는 값 없는 --resume) 을 뺀다."""
    out, skip = [], False
    for i, tok in enumerate(cmd):
        if skip:
            skip = False
            continue
        if tok in flags:
            has_value = tok != "--resume" and i + 1 < len(cmd) and not cmd[i + 1].startswith("--")
            skip = has_value
            continue
        out.append(tok)
    return out


def restart(job_id: str, changes: dict | None = None, resume_from: int | None = None) -> str:
    """(살아 있으면 먼저 종료하고) 설정을 바꿔서 체크포인트부터 다시 띄운다.
    changes 없으면 같은 설정으로 재시작. resume_from 없으면 마지막 체크포인트부터."""
    changes, resume_from = validate_changes(job_id, changes, resume_from)
    job = read_job(job_id)
    notes = []
    if is_alive(job["pid"], job_id):
        notes.append(terminate(job_id))

    flags = {"--" + k.replace("_", "-") for k in changes} | {"--resume", "--resume-from"}
    cmd = _without_flags(job["cmd"], flags)
    for k, v in changes.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    cmd += ["--resume-from", str(resume_from)] if resume_from is not None else ["--resume"]

    # 무엇을 왜 바꿨는지 기록 (나중에 "검증된 조치"를 skill로 저장할 때 쓴다)
    hist_path = os.path.join(job_dir(job_id), "changes.json")
    history = json.load(open(hist_path)) if os.path.exists(hist_path) else []
    history.append({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "changes": changes, "resume_from": resume_from,
                    "config_before": read_config(job_id)})
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    log = open(os.path.join(job_dir(job_id), "restart.out"), "a")
    p = subprocess.Popen([job.get("python", "python")] + cmd, cwd=job.get("cwd", ROOT), stdout=log,
                         stderr=subprocess.STDOUT, start_new_session=True)  # 우리 프로세스가 끝나도 job은 계속 돌게
    where = f"step {resume_from}" if resume_from is not None else "마지막 체크포인트"
    notes.append(f"{job_id}: {where}부터 재시작, 변경 {changes or '없음'} (새 pid {p.pid})")
    return " / ".join(notes)
