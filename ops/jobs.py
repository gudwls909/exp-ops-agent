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
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 테스트에서는 환경변수로 임시 폴더를 가리키게 한다 (week1 db.py의 DB_PATH와 같은 방식)
RUNS_DIR = os.getenv("OPS_RUNS_DIR", os.path.join(ROOT, "runs"))

JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
STEP_RE = re.compile(r"step (\d+)/(\d+)")


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


def restart(job_id: str) -> str:
    """죽은 job을 같은 명령 + --resume으로 다시 띄운다. 살아 있으면 거부."""
    job = read_job(job_id)
    if is_alive(job["pid"], job_id):
        return f"{job_id}: 아직 살아 있어서 재시작하지 않음 (먼저 terminate 필요)"
    cmd = [job.get("python", "python")] + job["cmd"] + (["--resume"] if "--resume" not in job["cmd"] else [])
    log = open(os.path.join(job_dir(job_id), "restart.out"), "a")
    p = subprocess.Popen(cmd, cwd=job.get("cwd", ROOT), stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)  # 우리 프로세스가 끝나도 job은 계속 돌게
    return f"{job_id}: --resume으로 재시작 (새 pid {p.pid})"
