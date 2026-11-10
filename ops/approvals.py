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
import runbooks

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


def request(job_id: str, action: str, reason: str, changes: dict | None = None,
            resume_from: int | None = None) -> dict:
    """승인 요청을 만든다. 실행하지 않는다.
    restart는 changes(바꿀 설정)와 resume_from(어느 체크포인트부터)을 함께 받을 수 있다.
    요청 시점에 검증한다 → 허용 범위 밖의 제안은 사람에게 가기 전에 거부된다."""
    jobs.job_dir(job_id)  # job_id 검증 (이상하면 ValueError)
    if action not in ACTIONS:
        raise ValueError(f"지원하지 않는 action: {action!r} (가능: {sorted(ACTIONS)})")
    if action == "restart":
        changes, resume_from = jobs.validate_changes(job_id, changes, resume_from)
        # "마지막 체크포인트"를 요청 시점의 구체적인 step으로 고정한다.
        # 승인을 기다리는 동안 job이 계속 돌며 새 체크포인트(망가진 것일 수도)를 쓰면,
        # 사람이 승인한 계획과 실제로 실행되는 계획이 달라진다 → "승인한 것 = 실행되는 것"
        if resume_from is None:
            steps = [c["step"] for c in jobs.checkpoints(job_id)]
            resume_from = steps[-1] if steps else None
    elif changes or resume_from is not None:
        raise ValueError("changes/resume_from은 restart에서만 쓸 수 있다")
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
        "changes": changes or {},
        "resume_from": resume_from,
        # 나중에 runbook으로 저장할 때 쓴다: 요청 시점의 장애 서명, 진단, 바꾸기 전 설정
        "diagnosis": reason[1:reason.index("]")] if reason.startswith("[") and "]" in reason else None,
        "signature": runbooks.signature(job_id),
        "config_before": jobs.read_config(job_id),
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
        # 실행 직전에 한 번 더 검증된다 (jobs.restart 안에서). 그 사이 상황이 바뀌어 실행할 수 없으면
        # 프로그램이 죽지 않고 "failed"와 이유를 남긴다 (평가 도중 실제로 죽었던 버그)
        try:
            if item["action"] == "terminate":
                item["result"] = jobs.terminate(item["job_id"])
            else:
                item["result"] = jobs.restart(item["job_id"], item.get("changes"), item.get("resume_from"))
            item["status"] = "executed"
        except ValueError as e:
            item["result"] = f"실행 직전 검증 실패: {e}"
            item["status"] = "failed"
    else:
        item["status"] = "rejected"
    item["decided_by"] = decided_by
    item["decided_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save(items)
    return item


def verify_outcomes() -> list[dict]:
    """실행된 조치의 결과를 확인한다 (Voyager의 critic 역할). 결과가 나온 것만 runbook에 기록.
    - restart: 재시작한 job이 끝까지 돌았나(Exit 0, loss 폭주 없음) / 다시 멈췄나 / 다시 죽었나
    - terminate: 프로세스가 사라졌나
    아직 돌고 있으면 다음에 다시 확인한다."""
    items = _load()
    newly = []
    for it in items:
        if it["status"] != "executed" or "outcome" in it:
            continue
        job_id = it["job_id"]
        try:
            pid = jobs.read_job(job_id)["pid"]
        except ValueError:
            continue
        alive = jobs.is_alive(pid, job_id)
        if it["action"] == "terminate":
            success, detail = (not alive), ("프로세스 종료 확인" if not alive else "아직 살아 있음")
        elif alive:
            if not jobs.is_stalled(job_id):
                continue  # 아직 정상적으로 도는 중 → 나중에 확인
            success, detail = False, "재시작 후 다시 멈춤"
        else:
            lines = jobs.tail(job_id, 100000)
            starts = [i for i, l in enumerate(lines) if "Start training" in l]
            seg = lines[starts[-1]:] if starts else lines
            sig_after = set(runbooks.signature(job_id))
            done = any("Exit 0" in l for l in seg)
            success = done and "loss_exploded" not in sig_after
            detail = "끝까지 정상 완료" if success else ("완료됐지만 loss 폭주" if done else "재시작 후 다시 죽음")
        it["outcome"] = {"success": success, "detail": detail, "checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        runbooks.record(it, success, detail)
        newly.append(it)
    if newly:
        _save(items)
    return newly
