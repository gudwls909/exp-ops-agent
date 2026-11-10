"""
Runbook 메모리 — "실제로 성공이 확인된 조치"만 저장하고, 비슷한 장애가 오면 꺼내 준다

Voyager의 skill library와 같은 원리
- Voyager: 환경에서 실행해 보고 critic이 성공을 확인한 코드만 skill로 저장 → 비슷한 태스크에서 검색해 재사용
- 여기:    사람이 승인해서 실행했고, 재시작한 job이 실제로 끝까지 돌았다고 확인된 조치만 runbook으로 저장
           (실패한 조치도 "이건 안 됐다"로 기록한다 → 같은 실수를 반복하지 않게. Reflexion에 가까운 부분)

검색 키 = 장애 "서명(signature)": 로그에서 코드로 뽑은 특징 토큰 묶음
  예) {"exc:OutOfMemoryError", "stage2_entered"}  /  {"silent_after_shard_read", "restarted_before"}
  임베딩 대신 토큰 집합 + Jaccard 유사도를 쓴다. 장애 로그는 정형화돼 있어서 이 정도로 충분하고,
  왜 그 runbook이 뽑혔는지 사람이 바로 읽을 수 있다. (토큰이 겹치지 않는 표현 차이가 많아지면 임베딩으로 바꾼다)

저장: state/runbooks.json. OPS_MEMORY=0이면 검색 결과를 항상 비우고 기록도 안 한다 (ablation용)
"""

import json
import math
import os
import re
import time

import jobs

STATE_DIR = os.getenv("OPS_RUNBOOK_DIR", os.getenv("OPS_STATE_DIR", os.path.join(jobs.ROOT, "state")))
EXC_RE = re.compile(r"\b([A-Z][A-Za-z]*Error)\b")
SHARD_READ_RE = re.compile(r"reading shard data/shard-(\d{4})")


def enabled() -> bool:
    return os.getenv("OPS_MEMORY", "1") != "0"


def _path():
    return os.path.join(STATE_DIR, "runbooks.json")


def _load() -> list[dict]:
    if not os.path.exists(_path()):
        return []
    with open(_path()) as f:
        return json.load(f)


def _save(items):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = _path() + ".tmp"
    with open(tmp, "w") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _path())


# ---------------------------------------------------------------
# 장애 서명: 로그에서 특징 토큰을 뽑는다 (LLM이 아니라 코드가 뽑는다 → 매번 같다)
# ---------------------------------------------------------------
def signature(job_id: str) -> list[str]:
    lines = jobs.tail(job_id, 100000)
    starts = [i for i, l in enumerate(lines) if "Start training" in l]
    seg = lines[starts[-1]:] if starts else lines  # 마지막 실행 구간
    text = "\n".join(seg)
    toks = {f"exc:{e}" for e in EXC_RE.findall(text)}
    if "corrupted shard" in text:
        toks.add("corrupted_shard")
    if "entering stage 2" in text:
        toks.add("stage2_entered")
    if "Training finished" in text:
        toks.add("training_finished")
    if len(starts) > 1:
        toks.add("restarted_before")
    # loss 폭주: 이 구간에서 loss가 nan/inf이거나 최저값의 100배 이상
    losses = []
    for l in seg:
        if (m := jobs.LOSS_RE.search(l)) and jobs.STEP_RE.search(l):
            try:
                losses.append(float(m.group(1)))
            except ValueError:
                pass
    finite = [v for v in losses if math.isfinite(v)]
    if losses and (len(finite) < len(losses) or (finite and max(finite) > 100 * max(min(finite), 1e-12))):
        toks.add("loss_exploded")
    # 마지막 의미 있는 줄이 "shard 읽는 중"이고 그 뒤로 아무것도 없음(또는 SIGTERM) → shard에서 멈춤
    meaningful = [l for l in seg if "Received SIGTERM" not in l and l.strip()]
    if meaningful and SHARD_READ_RE.search(meaningful[-1]):
        toks.add("silent_after_shard_read")
    return sorted(toks)


def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


# ---------------------------------------------------------------
# 조치에서 "다른 job에도 쓸 수 있는 규칙"을 뽑는다
#   숫자를 그대로 외우는 게 아니라 관계를 저장해야 다른 job(다른 batch, 다른 shard 번호)에 적용된다
# ---------------------------------------------------------------
def _lesson(job_id: str, changes: dict, config_before: dict) -> list[str]:
    out = []
    log = "\n".join(jobs.tail(job_id, 100000))
    if "data_shards" in changes:
        old = {int(x) for x in str(config_before.get("data_shards", "")).split(",") if x != ""}
        new = {int(x) for x in str(changes["data_shards"]).split(",") if x != ""}
        removed = sorted(old - new)
        read = {int(x) for x in SHARD_READ_RE.findall(log)}
        if all(k in read for k in removed):
            out.append(f"멈추거나 손상된 shard(로그에 마지막으로 읽은/손상으로 찍힌 shard)를 data_shards에서 뺐다 (이번엔 {removed})")
    if "grad_accum" in changes:
        big = max(int(config_before.get("batch_size", 1)), int(config_before.get("stage2_batch", 1)))
        micro = big // int(changes["grad_accum"])
        out.append(f"batch는 그대로 두고 grad_accum={changes['grad_accum']}로 한 번에 올라가는 batch를 {micro}로 줄였다 "
                   f"(이 GPU에서 micro-batch {micro}는 메모리에 들어갔다)")
    for k in ("stage2_lr", "max_lr", "warmup_steps"):
        if k in changes:
            out.append(f"{k}를 {config_before.get(k)} → {changes[k]}로 바꿨다")
    if not changes:
        out.append("설정 변경 없이 재시작했다")
    return out


def record(item: dict, success: bool, detail: str) -> dict | None:
    """실행된 조치의 결과를 기록한다. 같은 진단 + 비슷한 서명의 runbook이 있으면 거기에 합친다."""
    if not enabled():
        return None
    diag = item.get("diagnosis") or "unknown"
    sig = item.get("signature") or []
    changes = item.get("changes") or {}
    if item.get("action") == "terminate":
        kinds, lesson = ["(종료만)"], ["재시작하지 않고 종료만 했다"]
    else:
        kinds = sorted(changes) or ["(변경 없음)"]
        lesson = _lesson(item["job_id"], changes, item.get("config_before") or {})
    books = _load()
    # 같은 진단, 같은 종류의 변경, 서명이 충분히 비슷하면 같은 runbook
    rb = next((b for b in books if b["diagnosis"] == diag and b["change_kinds"] == kinds
               and _jaccard(b["signature"], sig) >= 0.6), None)
    if rb is None:
        rb = {"id": max((b["id"] for b in books), default=0) + 1, "diagnosis": diag, "signature": sig,
              "change_kinds": kinds, "successes": 0, "failures": 0, "examples": []}
        books.append(rb)
    rb["successes" if success else "failures"] += 1
    rb["examples"] = (rb["examples"] + [{
        "at": time.strftime("%Y-%m-%d %H:%M:%S"), "job_id": item["job_id"], "success": success,
        "changes": changes, "resume_from": item.get("resume_from"), "lesson": lesson, "detail": detail,
    }])[-5:]
    _save(books)
    return rb


def find(job_id: str, k: int = 3) -> list[dict]:
    """이 job의 서명과 비슷한 runbook. 성공/실패 횟수와 최근 사례(무엇을 바꿨고 결과가 어땠나)를 함께."""
    if not enabled():
        return []
    sig = signature(job_id)
    scored = sorted(((_jaccard(sig, b["signature"]), b) for b in _load()), key=lambda x: -x[0])
    out = []
    for score, b in scored[:k]:
        if score < 0.3:
            continue
        out.append({
            "runbook_id": b["id"], "similarity": round(score, 2), "diagnosis": b["diagnosis"],
            "signature": b["signature"],
            "action": "terminate" if b["change_kinds"] == ["(종료만)"] else "restart",
            "change_kinds": b["change_kinds"],
            "verified_successes": b["successes"], "failures": b["failures"],
            "recent": [{"success": e["success"], "changes": e["changes"], "lesson": e["lesson"]}
                       for e in b["examples"][-3:]],
        })
    return out
