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
    # 읽기 전용만. 요청도 도구가 아님
    assert tools == {"list_jobs", "job_status", "tail_log", "gpu_status", "job_config", "list_checkpoints",
                     "find_runbooks"}
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


# ---------------- 8. 설정 변경 제안의 검증 (허용 목록 + 범위) ----------------
def _job_with_config(tmp_path, job_id, ckpt_steps=(20, 40)):
    d = tmp_path / "runs" / job_id
    (d / "config.json").write_text(json.dumps({"max_lr": 1e-3, "stage2_lr": 1000.0, "grad_accum": 1,
                                                "stage2_batch": 1048576, "data_shards": "0,1,2,3"}))
    for st in ckpt_steps:
        (d / f"ckpt-{st:06d}.pt").write_bytes(b"x")
    return d


@pytest.mark.parametrize("changes,resume", [
    ({"steps": 10}, None),              # 실험 정체를 바꾸는 설정은 목록에 없음
    ({"seed": 1}, None),
    ({"stage2_lr": 5.0}, None),         # 범위 밖 (lr ≤ 1)
    ({"grad_accum": 0}, None),
    ({"grad_accum": "many"}, None),     # 타입 이상
    ({"data_shards": "0,1,2,3,9"}, None),  # shard 추가는 불가 (빼기만)
    ({"data_shards": ""}, None),        # 다 빼는 것도 불가
    ({}, 999),                          # 없는 체크포인트
    ({"stage2_batch": 1024}, None),     # batch는 바꿀 수 없다 (실험 의미 보존을 코드로 강제)
    ({"batch_size": 128}, None),
    ({"grad_accum": 2 ** 21}, None),    # batch보다 큰 grad_accum
])
def test_bad_changes_rejected(env, tmp_path, changes, resume):
    env("j1")
    _job_with_config(tmp_path, "j1")
    with pytest.raises(ValueError):
        approvals.request("j1", "restart", "x", changes=changes, resume_from=resume)
    assert approvals.pending() == []  # 거부된 제안은 사람에게 가지도 않는다


def test_changes_only_for_restart(env, tmp_path):
    env("j1")
    _job_with_config(tmp_path, "j1")
    with pytest.raises(ValueError):
        approvals.request("j1", "terminate", "x", changes={"max_lr": 1e-4})


def test_restart_with_changes_terminates_then_relaunches(env, tmp_path):
    # 가짜 job의 명령을 "인자를 무시하고 잠만 자는 파이썬"으로 → 재시작된 프로세스의 명령줄만 검사
    pid = env("j1")
    d = _job_with_config(tmp_path, "j1")
    job = json.loads((d / "job.json").read_text())
    job["cmd"] = ["-c", SLEEPER, "j1", "--stage2-lr", "1000", "--resume"]
    (d / "job.json").write_text(json.dumps(job))

    item = approvals.request("j1", "restart", "발산", changes={"stage2_lr": 1e-4, "grad_accum": None}, resume_from=20)
    assert item["changes"] == {"stage2_lr": 1e-4} and item["resume_from"] == 20
    done = approvals.decide(item["id"], approve=True)

    assert not alive(pid)  # 살아 있던 원래 프로세스는 먼저 종료됐고
    new_pid = int(done["result"].split("새 pid ")[1].rstrip(")"))
    try:
        cmdline = open(f"/proc/{new_pid}/cmdline", "rb").read().split(b"\0")
        args = [a.decode() for a in cmdline if a]
        assert args.count("--stage2-lr") == 1 and args[args.index("--stage2-lr") + 1] == "0.0001"  # 옛 값은 지워지고
        assert "--resume" not in args and args[args.index("--resume-from") + 1] == "20"
        history = json.loads((d / "changes.json").read_text())
        assert history[-1]["changes"] == {"stage2_lr": 1e-4}  # 무엇을 바꿨는지 기록
    finally:
        os.kill(new_pid, signal.SIGKILL)


# ---------------- 9. 검증 실패 이유를 Agent에게 돌려줄 수 있다 ----------------
def test_validation_errors_for_revision(env, tmp_path):
    import agent
    env("j1"); env("j2")
    _job_with_config(tmp_path, "j1"); _job_with_config(tmp_path, "j2")
    nulls = {k: None for k in ["max_lr", "stage2_lr", "warmup_steps", "grad_accum", "data_shards"]}
    report = {"jobs": [
        {"job_id": "j1", "action_requested": "restart", "changes": {**nulls, "grad_accum": 2 ** 21}, "resume_from_step": None},
        {"job_id": "j2", "action_requested": "restart", "changes": {**nulls, "grad_accum": 4}, "resume_from_step": 20},
    ]}
    errors = agent.validation_errors(report)
    assert list(errors) == ["j1"] and "grad_accum" in errors["j1"]  # 이유가 구체적이어야 고쳐 쓸 수 있다
    assert approvals.pending() == []  # 검증만 하고 대기열에는 넣지 않는다


def test_broken_checkpoint_rejected(env, tmp_path):
    env("j1")
    d = _job_with_config(tmp_path, "j1", ckpt_steps=(100, 150))
    # step 150에서 lr 폭주: 150 로그의 loss는 정상(업데이트 전 값)이지만 다음 로그가 폭주 → 체크포인트 150은 망가짐
    (d / "train.log").write_text(
        "2026-11-10 12:00:00 INFO step 100/300 loss 0.03\n"
        "2026-11-10 12:00:01 INFO step 150/300 loss 0.025\n"
        "2026-11-10 12:00:02 INFO step 160/300 loss 6e+31\n")
    with pytest.raises(ValueError, match="망가진"):
        approvals.request("j1", "restart", "x", changes={"stage2_lr": 1e-3}, resume_from=150)
    with pytest.raises(ValueError, match="망가진"):
        approvals.request("j1", "restart", "x", changes={"stage2_lr": 1e-3})  # 지정 안 하면 마지막(150) → 거부
    assert approvals.request("j1", "restart", "x", changes={"stage2_lr": 1e-3}, resume_from=100)["resume_from"] == 100


# ---------------- 10. 근거 대조: 다른 job의 진단을 베끼면 거부 ----------------
def test_grounding_rejects_evidence_not_in_this_log(env, tmp_path):
    import agent
    env("oomjob")
    d = _job_with_config(tmp_path, "oomjob")
    (d / "train.log").write_text("2026-11-10 12:00:00 INFO step 10/100 loss 0.5\n"
                                 "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.00 GiB\n")
    copied = 'KeyError "\'label\' missing (corrupted shard data/shard-0003.tar)"'   # 다른 job의 로그 내용
    assert "KeyError" in agent.grounding_error("oomjob", copied)
    assert agent.grounding_error("oomjob", "torch.OutOfMemoryError 발생") is None


def test_shard_removal_needs_proof_in_log(env, tmp_path):
    env("j1")
    d = _job_with_config(tmp_path, "j1")
    (d / "train.log").write_text("2026-11-10 12:00:00 INFO step 10/100 loss 0.5\n")
    with pytest.raises(ValueError, match="증거"):
        approvals.request("j1", "restart", "x", changes={"data_shards": "0,1,2"})  # shard 3 손상 증거 없음
    with open(d / "train.log", "a") as f:
        f.write("KeyError: \"'label' missing in sample index 3879 (corrupted shard data/shard-0003.tar)\"\n")
    assert approvals.request("j1", "restart", "x", changes={"data_shards": "0,1,2"})["changes"] == {"data_shards": "0,1,2"}


# ---------------- 11. 승인한 것 = 실행되는 것, 실행 불가면 죽지 않고 failed ----------------
def test_restart_pins_checkpoint_and_fails_gracefully(env, tmp_path):
    env("j1")
    d = _job_with_config(tmp_path, "j1", ckpt_steps=(20,))
    item = approvals.request("j1", "restart", "x")
    assert item["resume_from"] == 20                     # "마지막"을 요청 시점의 step으로 고정
    (d / "ckpt-000020.pt").unlink()                       # 승인 대기 중에 상황이 바뀜 (체크포인트가 사라짐)
    done = approvals.decide(item["id"], approve=True)
    assert done["status"] == "failed" and "검증 실패" in done["result"]
