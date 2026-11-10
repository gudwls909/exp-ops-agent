"""
승인 API — ops/approve.py(터미널)를 HTTP로. 웹 화면·슬랙 버튼·사내 시스템이 여기에 붙는다

  GET  /health                     상태 확인
  GET  /jobs                       job 요약 목록 (Agent가 보는 것과 같은 정보)
  POST /patrol                     Agent 순찰 1회 → 보고서 + 승인 요청 등록       🔒
  GET  /approvals?status=pending   승인 요청 목록 (무엇을, 왜, 어느 체크포인트부터)
  POST /approvals/{id}/approve     승인 → 이때 처음으로 실제 종료·재시작이 일어난다  🔒
  POST /approvals/{id}/reject      거절                                          🔒
  GET  /runbooks                   검증된 조치 기록 (메모리)

🔒 = X-API-Token 헤더 필요 (환경변수 OPS_API_TOKEN). 승인 API는 "학습 job을 죽일 수 있는 버튼"이라
     토큰 없이 열어 두면 네트워크에 접근하는 누구나 실험을 죽일 수 있다.
     (week1 Docs Agent의 TODO였던 "인증"을 여기서 처음 넣었다. 실서비스라면 사람별 계정 + 권한이 필요하다)

실행 (프로젝트 루트에서):
  OPS_API_TOKEN=... uvicorn api.app:app --port 8100
"""

import hmac
import os
import sys
import threading

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ops"))
sys.path.insert(0, os.path.join(ROOT, "agent"))

import agent  # noqa: E402
import approvals  # noqa: E402
import jobs  # noqa: E402
import runbooks  # noqa: E402

app = FastAPI(title="Experiment Ops Agent API")

# 승인 대기열은 JSON 파일 하나다. 요청 두 개가 동시에 승인하면 둘 다 "pending"을 읽고 둘 다 실행할 수 있다
# (같은 job을 두 번 재시작). 파일을 읽고 쓰는 구간을 한 번에 하나만 들어가게 잠근다.
# 서버 프로세스가 여러 개면 이 잠금으로는 부족하다 → 그때는 DB 트랜잭션(week1의 SQLite 등)으로 바꾼다
QUEUE_LOCK = threading.Lock()


def require_token(x_api_token: str = Header(default="")):
    expected = os.getenv("OPS_API_TOKEN", "")
    if not expected:
        raise HTTPException(503, "서버에 OPS_API_TOKEN이 설정되지 않아 조작 API를 쓸 수 없습니다")
    # compare_digest: 글자를 하나씩 비교하다 틀린 곳에서 바로 멈추면, 응답 시간으로 토큰을 한 글자씩 알아낼 수 있다
    if not hmac.compare_digest(x_api_token, expected):
        raise HTTPException(401, "토큰이 없거나 틀렸습니다")


class Decision(BaseModel):
    decided_by: str  # 누가 승인/거절했나 (감사 기록)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/jobs")
def list_jobs():
    mem = jobs.gpu_mem_by_pid()
    return [jobs.summary(j, mem) for j in jobs.list_job_ids()]


@app.post("/patrol", dependencies=[Depends(require_token)])
async def patrol():
    # Agent 순찰은 LLM 호출이라 수십 초 걸린다. async라서 그동안 다른 요청(/jobs 등)은 계속 처리된다
    result = await agent.run_once()
    return {"report": result["report"], "requests": result.get("requests", []),
            "llm_calls": result["trace"]["llm_calls"], "latency_s": result["trace"]["latency_s"]}


@app.get("/approvals")
def list_approvals(status: str | None = None):
    items = approvals._load()
    return [it for it in items if status is None or it["status"] == status]


def _decide(approval_id: int, approve: bool, who: str):
    # def(동기) 엔드포인트 → FastAPI가 스레드에서 실행한다. 종료는 SIGTERM 후 최대 15초 기다리므로
    # async로 두면 그동안 서버 전체가 멈춘다
    with QUEUE_LOCK:
        try:
            return approvals.decide(approval_id, approve=approve, decided_by=who)
        except ValueError as e:  # 없는 요청, 이미 처리된 요청
            raise HTTPException(409 if "이미" in str(e) else 404, str(e))


@app.post("/approvals/{approval_id}/approve", dependencies=[Depends(require_token)])
def approve(approval_id: int, body: Decision):
    return _decide(approval_id, True, body.decided_by)


@app.post("/approvals/{approval_id}/reject", dependencies=[Depends(require_token)])
def reject(approval_id: int, body: Decision):
    return _decide(approval_id, False, body.decided_by)


@app.get("/runbooks")
def list_runbooks():
    return runbooks._load()
