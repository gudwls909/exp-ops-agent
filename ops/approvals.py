"""
승인 대기열 — Agent는 "요청"만 할 수 있고, 실행은 사람이 승인해야 일어난다

흐름
  Agent(LLM) --request_action 도구--> request()  : 대기열에 pending으로 기록만 한다. 아무것도 죽이지 않는다
  사람        --approve.py / 나중에 API--> decide() : 승인하면 그때 jobs.terminate / jobs.restart 실행

왜 이렇게 나누나
- LLM의 판단은 확률적이다 (week1 평가에서 같은 공격도 1/3은 도구를 불렀다)
- 실험을 잘못 죽이는 비용이 크다 → 되돌릴 수 없는 조작은 사람이 마지막에 확인
- "LLM이 kill 도구를 안 부르길 기대"가 아니라, 애초에 LLM에게 kill 도구를 주지 않는다 (구조로 막기)

저장: 연습 단계라 JSON 파일 하나. (FastAPI 단계에서 week1처럼 SQLite로 바꾼다)
"""

import json
import os
import time

import jobs

STATE_DIR = os.getenv("OPS_STATE_DIR", os.path.join(jobs.ROOT, "state"))
ACTIONS = {"terminate", "restart"}


def _path():
    return os.path.join(STATE_DIR, "approvals.json")


def _load() -> list[dict]:
    if not os.path.exists(_path()):
        return []
    with open(_path()) as f:
        return json.load(f)


def _save(items: list[dict]) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = _path() + ".tmp"
    with open(tmp, "w") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _path())  # 쓰다가 죽어도 파일이 반쯤 깨지지 않게: 임시 파일에 다 쓴 뒤 한 번에 교체


def request(job_id: str, action: str, reason: str) -> dict:
    """승인 요청을 만든다. 실행하지 않는다."""
    jobs.job_dir(job_id)  # job_id 검증 (이상하면 ValueError)
    if action not in ACTIONS:
        raise ValueError(f"지원하지 않는 action: {action!r} (가능: {sorted(ACTIONS)})")
    items = _load()
    # 같은 job, 같은 action으로 이미 대기 중인 요청이 있으면 새로 만들지 않는다 (Agent가 같은 요청을 반복하는 경우)
    for it in items:
        if it["job_id"] == job_id and it["action"] == action and it["status"] == "pending":
            return it
    item = {
        "id": max((it["id"] for it in items), default=0) + 1,
        "job_id": job_id,
        "action": action,
        "reason": reason,
        "status": "pending",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    items.append(item)
    _save(items)
    return item


def pending() -> list[dict]:
    return [it for it in _load() if it["status"] == "pending"]


def decide(approval_id: int, approve: bool, decided_by: str = "human") -> dict:
    """사람의 결정. 승인이면 여기서 처음으로 실제 조작이 일어난다."""
    items = _load()
    item = next((it for it in items if it["id"] == approval_id), None)
    if item is None:
        raise ValueError(f"없는 승인 요청: {approval_id}")
    if item["status"] != "pending":
        raise ValueError(f"이미 처리된 요청: {approval_id} ({item['status']})")

    if approve:
        run = jobs.terminate if item["action"] == "terminate" else jobs.restart
        item["result"] = run(item["job_id"])
        item["status"] = "executed"
    else:
        item["status"] = "rejected"
    item["decided_by"] = decided_by
    item["decided_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save(items)
    return item
