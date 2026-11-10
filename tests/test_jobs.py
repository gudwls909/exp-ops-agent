"""요약 신호 테스트: 로그 파싱이 맞아야 Agent도 규칙도 맞게 판단한다"""

import json
import time

import jobs

LOG = """2026-11-10 12:00:00 INFO Start training job=j1
2026-11-10 12:00:01 INFO step 10/300 loss 0.8436 lr 1.0e-03 gpu_mem 97MiB step_time 0.10s
2026-11-10 12:00:02 INFO step 20/300 loss 0.8009 lr 1.0e-03 gpu_mem 97MiB step_time 0.10s
2026-11-10 12:00:03 INFO step 30/300 loss 0.6329 lr 1.0e-03 gpu_mem 97MiB step_time 0.10s
2026-11-10 12:00:04 INFO Saved checkpoint at step 30
2026-11-10 12:00:23 INFO step 40/300 loss nan lr 1.0e+12 gpu_mem 97MiB step_time 2.00s
"""


def make(tmp_path, monkeypatch, log):
    monkeypatch.setattr(jobs, "RUNS_DIR", str(tmp_path))
    (tmp_path / "j1").mkdir()
    (tmp_path / "j1" / "job.json").write_text(json.dumps({"job_id": "j1", "pid": 999999999, "cmd": []}))
    (tmp_path / "j1" / "train.log").write_text(log)


def test_last_metrics(tmp_path, monkeypatch):
    make(tmp_path, monkeypatch, LOG)
    assert jobs.last_step("j1") == (40, 300)
    assert jobs.last_metrics("j1") == ("nan", 2.0)  # loss는 글자 그대로


def test_recent_log_interval_is_median(tmp_path, monkeypatch):
    make(tmp_path, monkeypatch, LOG)
    # 간격: 1, 1, 20초 → 중앙값 1초 (한 번 튄 간격에 휘둘리지 않는다)
    assert jobs.recent_log_interval("j1") == 1.0


def test_no_step_lines(tmp_path, monkeypatch):
    make(tmp_path, monkeypatch, "2026-11-10 12:00:00 INFO Start training\n")
    s = jobs.summary("j1", gpu_mem={})
    assert s["last_step"] is None and s["recent_log_interval_s"] is None and s["log_age_vs_interval"] is None
