"""승인 API 테스트: 토큰 없이는 아무것도 조작할 수 없고, 승인했을 때만 실행되며, 두 번 승인되지 않는다"""

import json
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

import approvals
import jobs
import runbooks
from api import app as api_app

SLEEPER = "import time\nwhile True: time.sleep(0.1)"
TOKEN = "test-token"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(approvals, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(runbooks, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPS_API_TOKEN", TOKEN)
    procs = []

    def make_job(job_id):
        p = subprocess.Popen([sys.executable, "-c", SLEEPER, job_id])
        procs.append(p)
        d = tmp_path / "runs" / job_id
        d.mkdir(parents=True)
        (d / "job.json").write_text(json.dumps({"job_id": job_id, "pid": p.pid, "cmd": ["--fail", "hang"]}))
        (d / "train.log").write_text("2026-11-10 12:00:00 INFO step 10/100 loss 0.5\n")
        time.sleep(0.2)
        return p

    yield TestClient(api_app.app), make_job
    for p in procs:
        if p.poll() is None:
            p.kill()
        p.wait()


def auth(who="tester"):
    return {"headers": {"X-API-Token": TOKEN}, "json": {"decided_by": who}}


def test_health_and_jobs_hide_answer(client):
    c, make_job = client
    make_job("j1")
    assert c.get("/health").json() == {"status": "ok"}
    body = c.get("/jobs").text
    assert "j1" in body and "--fail" not in body and "hang" not in body  # 정답(cmd) 누출 없음


def test_no_token_no_action(client, monkeypatch):
    c, make_job = client
    p = make_job("j1")
    item = approvals.request("j1", "terminate", "[hung] x")
    assert c.post(f"/approvals/{item['id']}/approve", json={"decided_by": "x"}).status_code == 401
    assert c.post(f"/approvals/{item['id']}/approve", headers={"X-API-Token": "wrong"},
                  json={"decided_by": "x"}).status_code == 401
    assert c.post("/patrol").status_code == 401
    monkeypatch.setenv("OPS_API_TOKEN", "")  # 서버에 토큰이 설정되지 않았으면 조작 API 자체를 막는다
    assert c.post(f"/approvals/{item['id']}/approve", **auth()).status_code == 503
    assert p.poll() is None  # 아무것도 죽지 않았다


def test_approve_flow_and_no_double_execution(client):
    c, make_job = client
    p = make_job("j1")
    item = approvals.request("j1", "terminate", "[hung] 로그 정지")
    pending = c.get("/approvals", params={"status": "pending"}).json()
    assert [it["id"] for it in pending] == [item["id"]]

    r = c.post(f"/approvals/{item['id']}/approve", **auth("alice"))
    assert r.status_code == 200 and r.json()["status"] == "executed" and r.json()["decided_by"] == "alice"
    p.wait(timeout=5)  # 실제로 종료됐다
    assert c.post(f"/approvals/{item['id']}/approve", **auth("bob")).status_code == 409  # 두 번 실행 안 됨
    assert c.post("/approvals/999/approve", **auth()).status_code == 404


def test_reject_keeps_job(client):
    c, make_job = client
    p = make_job("j1")
    item = approvals.request("j1", "terminate", "[hung] x")
    assert c.post(f"/approvals/{item['id']}/reject", **auth()).json()["status"] == "rejected"
    assert p.poll() is None


def test_patrol_endpoint(client, monkeypatch):
    c, _ = client

    async def fake_run_once():
        return {"report": {"jobs": []}, "requests": [], "trace": {"llm_calls": 1, "latency_s": 0.1}}

    monkeypatch.setattr(api_app.agent, "run_once", fake_run_once)
    r = c.post("/patrol", headers={"X-API-Token": TOKEN})
    assert r.status_code == 200 and r.json()["llm_calls"] == 1


def test_os_error_is_recorded_not_crashed(client, monkeypatch):
    # 컨테이너 보안 정책 등으로 신호를 못 보내는 경우 → 500 대신 "failed"와 이유가 남는다
    c, make_job = client
    make_job("j1")
    item = approvals.request("j1", "terminate", "[hung] x")

    def denied(job_id, grace_s=15.0):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(jobs, "terminate", denied)
    r = c.post(f"/approvals/{item['id']}/approve", **auth())
    assert r.status_code == 200 and r.json()["status"] == "failed" and "권한" in r.json()["result"]
