"""Runbook 메모리 테스트: 서명 추출, 검증된 조치만 기록, 검색, 결과 확인(critic), 메모리 끄기"""

import json

import pytest

import approvals
import jobs
import runbooks

T = "2026-11-10 12:00:{:02d} INFO "
OOM_LOG = "\n".join([
    T.format(0) + "Start training job=j device=cuda:1 steps=150 pid=1",
    T.format(1) + "step 110/150 loss 0.03 lr 1.0e-03 gpu_mem 97MiB step_time 0.05s",
    T.format(2) + "curriculum: entering stage 2, batch_size=524288 grad_accum=1 lr=1.0e-03",
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.00 GiB",
]) + "\n"
STALL_LOG = "\n".join([
    T.format(0) + "Start training job=j device=cuda:1 steps=300 pid=1",
    T.format(1) + "step 80/300 loss 0.03 lr 1.0e-03 gpu_mem 97MiB step_time 0.05s",
    T.format(2) + "data: reading shard data/shard-0002.tar",
    "2026-11-10 12:00:40 WARNING Received SIGTERM at step 81, saving checkpoint and exiting",
    T.format(41) + "Resumed from checkpoint at step 81",
    T.format(42) + "Start training job=j device=cuda:1 steps=300 pid=2",
    T.format(43) + "data: reading shard data/shard-0002.tar",
]) + "\n"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(approvals, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(runbooks, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPS_MEMORY", "1")

    def make(job_id, log, config=None, ckpts=(40, 80)):
        d = tmp_path / "runs" / job_id
        d.mkdir(parents=True)
        (d / "job.json").write_text(json.dumps({"job_id": job_id, "pid": 999999999, "cmd": []}))  # 죽은 pid
        (d / "train.log").write_text(log)
        (d / "config.json").write_text(json.dumps(config or {"batch_size": 256, "stage2_batch": 524288,
                                                             "grad_accum": 1, "data_shards": "0,1,2,3"}))
        for st in ckpts:
            (d / f"ckpt-{st:06d}.pt").write_bytes(b"x")
        return d
    return make


def test_signature_is_from_code_not_llm(env):
    env("oom", OOM_LOG)
    env("stall", STALL_LOG)
    assert runbooks.signature("oom") == ["exc:OutOfMemoryError", "stage2_entered"]
    assert set(runbooks.signature("stall")) == {"silent_after_shard_read", "restarted_before"}


def test_stall_shard_counts_as_evidence(env):
    env("stall", STALL_LOG)
    # 지난 실행이 shard 2를 읽다가 멈춘 채 끝났다 → shard 2를 뺄 증거가 있다
    assert jobs.stall_shards("stall") == {2}
    item = approvals.request("stall", "restart", "[hung] shard 2에서 반복 멈춤", changes={"data_shards": "0,1,3"})
    assert item["changes"] == {"data_shards": "0,1,3"}
    with pytest.raises(ValueError, match="증거"):  # 증거 없는 shard 1은 못 뺀다
        approvals.request("stall", "restart", "x", changes={"data_shards": "0,2,3"})


def _executed(job_id, changes, diagnosis):
    item = approvals.request(job_id, "restart", f"[{diagnosis}] test", changes=changes)
    items = approvals._load()
    for it in items:
        if it["id"] == item["id"]:
            it["status"] = "executed"   # 실제 재시작 대신 "실행됐다"고만 표시 (결과는 로그로 판정)
    approvals._save(items)
    return item


def test_only_verified_success_becomes_reusable(env):
    d = env("oom", OOM_LOG)
    _executed("oom", {"grad_accum": 4}, "oom")
    # 아직 결과를 모름 → 재시작 로그를 붙여서 "끝까지 돌았다"를 만든다
    with open(d / "train.log", "a") as f:
        f.write(T.format(50) + "Start training job=oom steps=150 pid=3\n")
        f.write(T.format(51) + "step 150/150 loss 0.02 lr 1.0e-03 gpu_mem 353MiB step_time 1.55s\n")
        f.write(T.format(52) + "Exit 0\n")
    done = approvals.verify_outcomes()
    assert done[0]["outcome"]["success"] is True

    # 다른 OOM job (다른 batch) → 검색하면 이 runbook이 나오고, 숫자가 아니라 관계(micro-batch)를 알려 준다
    env("oom2", OOM_LOG.replace("524288", "786432"), config={"batch_size": 512, "stage2_batch": 786432,
                                                             "grad_accum": 1, "data_shards": "0,1"})
    found = runbooks.find("oom2")
    assert found and found[0]["verified_successes"] == 1 and found[0]["change_kinds"] == ["grad_accum"]
    assert "micro-batch 131072" in found[0]["recent"][0]["lesson"][0]


def test_failed_fix_is_recorded_as_failure(env):
    d = env("stall", STALL_LOG)
    _executed("stall", {}, "hung")  # 같은 설정으로 재시작
    with open(d / "train.log", "a") as f:  # 재시작했지만 같은 shard에서 또 멈추고 종료됨
        f.write(T.format(55) + "Start training job=stall steps=300 pid=4\n")
        f.write(T.format(56) + "data: reading shard data/shard-0002.tar\n")
        f.write("2026-11-10 12:01:30 WARNING Received SIGTERM at step 82, saving checkpoint and exiting\n")
    approvals.verify_outcomes()
    rb = runbooks._load()[0]
    assert rb["failures"] == 1 and rb["successes"] == 0  # "이건 안 됐다"도 기억한다


def test_memory_off_for_ablation(env, monkeypatch):
    env("oom", OOM_LOG)
    monkeypatch.setenv("OPS_MEMORY", "0")
    assert runbooks.record({"job_id": "oom", "changes": {}}, True, "x") is None
    assert runbooks.find("oom") == []


def test_slow_after_restart_is_not_stalled(env, monkeypatch):
    # grad_accum을 늘려 재시작 → step이 1.5초로 느려짐. 예전의 빠른 간격(1초 미만)과 비교하면 안 된다
    import time as _t
    now = _t.time()
    fmt = lambda t: _t.strftime("%Y-%m-%d %H:%M:%S", _t.localtime(t))  # noqa: E731
    log = "\n".join([
        f"{fmt(now - 100)} INFO Start training job=j steps=110 pid=1",
        f"{fmt(now - 99)} INFO step 10/110 loss 0.5 lr 1e-3 gpu_mem 9MiB step_time 0.03s",
        f"{fmt(now - 98)} INFO step 20/110 loss 0.4 lr 1e-3 gpu_mem 9MiB step_time 0.03s",
        f"{fmt(now - 60)} INFO Start training job=j steps=110 pid=2",
        f"{fmt(now - 40)} INFO step 100/110 loss 0.03 lr 1e-3 gpu_mem 9MiB step_time 1.55s",
    ]) + "\n"
    d = env("j", log)
    monkeypatch.setattr(jobs, "is_alive", lambda pid, job_id: True)
    import os
    os.utime(d / "train.log", (now - 20, now - 20))  # 마지막 로그 20초 전 (예상 간격 15.5초 → 정상)
    assert jobs.is_stalled("j") is False
    os.utime(d / "train.log", (now - 200, now - 200))  # 200초 조용 → 멈춤
    assert jobs.is_stalled("j") is True


def test_memory_blocks_repeating_a_failed_action(env):
    import agent
    env("old", STALL_LOG)
    for _ in range(2):  # 같은 장애에서 "변경 없이 재시작"이 2번 실패한 기록
        runbooks.record({"job_id": "old", "action": "restart", "diagnosis": "hung", "changes": {},
                         "signature": ["silent_after_shard_read"]}, False, "재시작 후 다시 멈춤")
    first_time = STALL_LOG.split("\n")[:3]  # 이 job은 처음 멈춤 (재시작 전)
    env("new", "\n".join(first_time) + "\n")
    report = {"jobs": [{"job_id": "new", "action_requested": "restart", "evidence": "", "resume_from_step": None,
                        "changes": {k: None for k in ["max_lr", "stage2_lr", "warmup_steps", "grad_accum", "data_shards"]}}]}
    errors = agent.validation_errors(report)
    assert "2번 실패" in errors["new"]
