"""
안전장치 테스트 — 이 프로젝트에서 가장 중요한 테스트

지키는 약속
1. Agent에게 주는 MCP 도구 중에 "실행" 도구는 없다 (요청 도구만 있다)
2. 요청만으로는 아무것도 죽지 않는다. 사람이 승인해야 실행된다
3. 이상한 job_id(경로 탐색)는 거부된다
4. pid가 재사용되어 엉뚱한 프로세스를 가리키면 죽이지 않는다
5. 종료는 SIGTERM 먼저, 반응이 없을 때만 SIGKILL
6. Agent가 보는 정보에 정답(--fail)이 새지 않는다

진짜 학습 job 대신 "sleep만 하는 작은 파이썬 프로세스"를 띄워서 빠르게 테스트한다.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time

import pytest

import approvals
import jobs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SLEEPER = "import sys, time\nwhile True: time.sleep(0.1)"
STUBBORN = "import signal, sys, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nwhile True: time.sleep(0.1)"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(approvals, "STATE_DIR", str(tmp_path / "state"))
    procs = []

    def make_job(job_id, code=SLEEPER, pid=None):
        """가짜 job: job_id를 명령줄 인자로 가진 프로세스 + job.json"""
        d = tmp_path / "runs" / job_id
        d.mkdir(parents=True)
        if pid is None:
            p = subprocess.Popen([sys.executable, "-c", code, job_id])
            procs.append(p)
            pid = p.pid
        (d / "job.json").write_text(json.dumps({
            "job_id": job_id, "pid": pid, "python": sys.executable,
            "cmd": ["sim/fake_train.py", "--job-id", job_id, "--fail", "hang"],  # 정답이 들어 있는 cmd
        }))
        (d / "train.log").write_text("2026-11-10 12:00:00 INFO step 10/100 loss 0.5\n")
        time.sleep(0.2)  # 프로세스가 뜰 시간
        return pid

    yield make_job
    for p in procs:  # 테스트가 실패해도 프로세스는 정리
        if p.poll() is None:
            p.kill()
        p.wait()


def alive(pid):
    # 테스트용 단순 확인 (좀비는 죽은 것으로)
    try:
        with open(f"/proc/{pid}/status") as f:
            return "Z" not in next(l for l in f if l.startswith("State:"))
    except FileNotFoundError:
        return False


# ---------------- 1. MCP 도구 목록에 실행 도구가 없다 ----------------
def test_mcp_exposes_no_execute_tools():
    from mcp import Client, StdioServerParameters

    async def names():
        server = StdioServerParameters(command=sys.executable, args=[os.path.join(ROOT, "ops", "server.py")])
        async with Client(server) as c:
            return {t.name for t in (await c.list_tools()).tools}

    tools = asyncio.run(names())
    assert tools == {"list_jobs", "job_status", "tail_log", "gpu_status"}  # 읽기 전용만. 요청도 도구가 아님
    # 혹시 누가 실행 도구를 추가하면 이 테스트가 깨진다 → 리뷰에서 "왜 추가했나" 논의하게 된다


# ---------------- 2. 요청만으로는 아무것도 안 죽는다 ----------------
def test_request_does_not_kill(env):
    pid = env("j1")
    item = approvals.request("j1", "terminate", "로그 정지")
    assert item["status"] == "pending"
    time.sleep(0.5)
    assert alive(pid)


def test_duplicate_request_is_merged(env):
    env("j1")
    a = approvals.request("j1", "terminate", "1")
    b = approvals.request("j1", "terminate", "2")
    assert a["id"] == b["id"] and len(approvals.pending()) == 1


@pytest.mark.parametrize("action", ["kill", "rm -rf", "delete", "TERMINATE"])
def test_unknown_action_rejected(env, action):
    env("j1")
    with pytest.raises(ValueError):
        approvals.request("j1", action, "x")


def test_approve_terminates(env):
    pid = env("j1")
    item = approvals.request("j1", "terminate", "로그 정지")
    done = approvals.decide(item["id"], approve=True)
    assert done["status"] == "executed" and "SIGTERM" in done["result"]
    assert not alive(pid)


def test_reject_keeps_job_and_cannot_decide_twice(env):
    pid = env("j1")
    item = approvals.request("j1", "terminate", "로그 정지")
    assert approvals.decide(item["id"], approve=False)["status"] == "rejected"
    assert alive(pid)
    with pytest.raises(ValueError):
        approvals.decide(item["id"], approve=True)  # 거절한 걸 나중에 승인으로 뒤집을 수 없다


# ---------------- 3. 경로 탐색 ----------------
@pytest.mark.parametrize("bad", ["../etc", "a/b", "..", "", "/tmp", "j1/../../x", "-rf"])
def test_bad_job_id_rejected(env, bad):
    env("j1")
    with pytest.raises(ValueError):
        jobs.job_dir(bad)
    with pytest.raises(ValueError):
        approvals.request(bad, "terminate", "x")


# ---------------- 4. pid 재사용 ----------------
def test_pid_reuse_is_not_killed(env):
    # job.json의 pid가 지금은 "다른 프로세스"(여기서는 pytest 자신)를 가리키는 상황
    env("old", pid=os.getpid())
    assert jobs.is_alive(os.getpid(), "old") is False  # 명령줄에 job_id가 없으니 우리 job이 아니다
    assert "이미 종료됨" in jobs.terminate("old")       # 그래서 죽이지 않는다 (죽였다면 pytest가 죽어서 이 줄에 못 온다)


# ---------------- 5. SIGTERM → SIGKILL ----------------
def test_sigkill_only_when_sigterm_ignored(env):
    pid = env("stubborn", code=STUBBORN)
    result = jobs.terminate("stubborn", grace_s=1)
    assert "SIGKILL" in result
    assert not alive(pid)


# ---------------- 6. 정답 누출 방지 ----------------
def test_agent_view_has_no_answer(env):
    env("j1")
    import server
    for out in (server.list_jobs(), server.job_status("j1"), server.tail_log("j1")):
        assert "--fail" not in out and "hang" not in out and '"cmd"' not in out


# ---------------- 7. 보고서(말) → 승인 대기열(행동)은 코드가 옮긴다 ----------------
def test_report_actions_are_filed_by_code(env):
    import agent
    env("j1"); env("j2")
    report = {"jobs": [
        {"job_id": "j1", "diagnosis": "hung", "evidence": "25배 지연", "action_requested": "terminate", "recommendation": ""},
        {"job_id": "j2", "diagnosis": "healthy", "evidence": "정상", "action_requested": "none", "recommendation": ""},
        {"job_id": "ghost", "diagnosis": "hung", "evidence": "지어낸 job", "action_requested": "terminate", "recommendation": ""},
    ]}
    filed = agent.file_requests(report)
    assert [it["job_id"] for it in approvals.pending()] == ["j1"]   # 요청한 것만, 정확히 하나
    assert "error" in next(f for f in filed if f["job_id"] == "ghost")  # 없는 job은 거부 (지어낸 id)
