"""
Ops Agent — MCP ops 서버에 붙어서 job들을 한 번 "순찰"하고, 진단 보고서와 승인 요청을 남긴다

week1 step3_agent.py / step5_api.py의 run_agent와 같은 루프다. 달라진 점
1. 시스템 프롬프트에 "운영 정책"이 들어간다 (언제 종료를 요청하고, 언제 건드리지 말아야 하나)
2. 마지막 답을 자유 텍스트가 아니라 JSON 보고서로 받는다 (Structured Outputs)
   → 사람이 읽기도 좋고, 평가 코드가 "진단이 맞았나"를 자동 채점할 수 있다 (week1 judge.py와 같은 기법)
3. 한 번의 순찰에 도구를 여러 번 부른다 (job 여러 개 × 상태 확인/로그 확인) → MAX_STEPS를 넉넉히

실행:
  python agent/agent.py            # 순찰 1회 → 보고서 출력 + 보고서의 조치를 승인 대기열에 등록
  python ops/approve.py            # 대기 중인 승인 요청 확인/승인
"""

import asyncio
import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from openai import AsyncOpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))
sys.path.insert(0, os.path.join(ROOT, "ops"))

import approvals  # noqa: E402
import jobs  # noqa: E402
import runbooks  # noqa: E402

MAX_REVISIONS = 1  # 검증에서 거부된 제안을 고쳐 쓸 기회

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
MAX_STEPS = 15


def ops_server() -> StdioServerParameters:
    # 함수로 만든 이유: import할 때 한 번 만들어 두면 그 순간의 환경변수(OPS_RUNS_DIR)가 고정된다.
    # 평가처럼 한 프로세스에서 여러 번 돌릴 때, 두 번째부터 엉뚱한 폴더를 보게 된다 (실제로 겪은 버그)
    return StdioServerParameters(command=sys.executable, args=[os.path.join(ROOT, "ops", "server.py")],
                                 env={**os.environ})

DIAGNOSES = ["healthy", "slow_but_progressing", "finished", "hung", "hung_at_shutdown",
             "crashed", "oom", "nan_loss", "unknown"]

SYSTEM_PROMPT = """너는 GPU 학습 서버를 돌보는 운영(ops) 엔지니어다. 모든 학습 job을 점검하고 진단 보고서를 쓴다.

점검 절차
1. list_jobs로 전체 상태를 본다.
2. 조금이라도 이상해 보이는 job(프로세스가 죽었음, 로그가 오래 안 바뀜, step이 total에 못 미친 채 멈춤 등)은
   반드시 tail_log로 로그를 직접 확인한 뒤에 진단한다. 숫자 요약만 보고 진단하지 마라.
   - "로그가 오래 안 바뀜"은 절대 시간이 아니라 log_age_vs_interval(평소 로그 간격 대비 배수)로 판단한다.
     예: 평소 1초마다 쓰던 job이 25초째 조용하면(25배) 멈춘 것이다. 평소 20초마다 쓰던 job이 15초째면 정상이다.
   - last_loss, last_step_time_s는 "마지막으로 찍힌 줄"의 값이다. 멈춘 job도 멈추기 직전 줄은 정상이므로,
     이 값이 정상이라고 해서 지금 진행 중이라는 뜻은 아니다.
3. 필요하면 gpu_status, job_status를 쓴다.
4. 재시작을 제안할 job은 job_config로 설정을 보고, list_checkpoints로 어느 체크포인트부터 다시 할지 고른다.
5. 조치(terminate/restart)를 제안하기 전에 find_runbooks로 과거에 실제로 검증된 조치를 확인한다.
   - 성공 기록이 있는 방법은 지금 job의 설정·로그에 맞게 적용한다 (숫자를 그대로 베끼지 말고 lesson의 관계를 적용)
   - 실패 기록이 있는 방법(예: 같은 설정으로 재시작했더니 같은 자리에서 또 멈춤)은 반복하지 않는다
   - 기록이 없거나 지금 장애와 맞지 않으면 무시하고 아래 정책대로 판단한다

진단 종류 (diagnosis)
- healthy              : 정상 진행 중
- slow_but_progressing : 느려졌지만 step이 늘고 loss가 정상 값으로 계속 찍힌다 → 정상. 절대 종료 요청하지 마라
- finished             : 정상 종료 (Training finished 후 프로세스 종료)
- hung                 : 학습 도중 멈춤 (프로세스는 살아 있는데 로그가 진행되지 않음)
- hung_at_shutdown     : 학습은 끝났다는 로그(Iteration limit reached 등)가 있는데 프로세스가 안 끝남
- crashed              : 예외(traceback)로 죽음 (OOM 제외)
- oom                  : CUDA out of memory로 죽음
- nan_loss             : loss가 nan/inf이거나 갑자기 폭주(예: 0.02 → 1e+31)했는데 계속 돌고 있음 (GPU 낭비)
- unknown              : 판단할 근거가 부족함

조치 정책 (보고서의 action_requested. 여기 적은 조치가 그대로 사람에게 승인 요청으로 간다)
- restart는 "살아 있으면 먼저 종료한 뒤, changes로 설정을 바꿔 resume_from_step 체크포인트부터 다시 실행"이다.
  원인을 고치지 않고 그대로 재시작하면 같은 실패가 반복된다. 원인이 설정에 있으면 반드시 그 설정을 고쳐라.
- hung             → restart. 보통 일시적 멈춤이므로 changes 없음, resume_from_step은 null(마지막 체크포인트).
                     단 이미 같은 설정으로 재시작했는데 같은 지점(예: 같은 shard를 읽다가)에서 또 멈췄다면,
                     원인을 그 지점으로 보고 고친다 (예: 그 shard를 data_shards에서 빼기)
- hung_at_shutdown → terminate (학습은 이미 끝났으므로 재시작 불필요)
- nan_loss         → restart. 원인 설정(로그에서 loss가 망가진 시점에 바뀐 lr 등)을 job_config에서 찾아 안전한 값으로 낮춘다.
                     resume_from_step은 list_checkpoints에서 loss가 아직 정상이던 마지막 체크포인트 (망가진 뒤 저장된 것은 쓰지 마라)
- oom              → restart. 실질 batch size는 유지하고 grad_accum을 늘려 한 번에 올라가는 batch를 줄인다
                     (batch 자체를 줄이면 실험 의미가 바뀐다). resume_from_step은 null
- crashed          → 원인이 손상된 데이터 shard면 restart로 그 shard를 data_shards에서 뺀다 (데이터가 바뀌므로
                     recommendation에 "사람 확인 필요"를 쓴다). 다른 원인이면 none으로 두고 원인만 보고
- healthy, slow_but_progressing, finished, unknown → none
- 원인과 관련된 설정만 최소한으로 바꾼다. 바꾸지 않을 설정은 changes에서 null로 둔다
- 실험을 잘못 죽이는 것이 가장 비싼 실수다. 확신이 없으면 요청하지 말고 unknown으로 보고하라
- evidence에는 사람이 승인 여부를 바로 판단할 수 있게 근거(로그 내용, 수치)를 쓴다

마지막 답은 보고서 JSON으로 한다. evidence와 recommendation은 한국어로 짧게."""

REPORT_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "ops_report",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "jobs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "job_id": {"type": "string"},
                            "diagnosis": {"type": "string", "enum": DIAGNOSES},
                            "evidence": {"type": "string"},
                            "action_requested": {"type": "string", "enum": ["none", "terminate", "restart"]},
                            # restart일 때만 의미 있음. 바꾸지 않는 설정은 null
                            "changes": {
                                "type": "object",
                                "properties": {
                                    "max_lr": {"type": ["number", "null"]},
                                    "stage2_lr": {"type": ["number", "null"]},
                                    "warmup_steps": {"type": ["integer", "null"]},
                                    "grad_accum": {"type": ["integer", "null"]},
                                    "data_shards": {"type": ["string", "null"], "description": "예: \"0,1,2,4\""},
                                },
                                "required": ["max_lr", "stage2_lr", "warmup_steps", "grad_accum", "data_shards"],
                                "additionalProperties": False,
                            },
                            "resume_from_step": {"type": ["integer", "null"]},
                            "recommendation": {"type": "string"},
                        },
                        "required": ["job_id", "diagnosis", "evidence", "action_requested", "changes",
                                     "resume_from_step", "recommendation"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["jobs"],
            "additionalProperties": False,
        },
    },
}


async def patrol(llm: AsyncOpenAI, mcp: Client, tools: list) -> dict:
    """순찰 1회. 보고서 + 실행 기록(도구 호출, 토큰, 시간)을 돌려준다."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "지금 모든 job을 점검하고 보고서를 써줘."},
    ]
    trace = {"tool_calls": [], "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "revisions": []}
    t0 = time.perf_counter()

    for _ in range(MAX_STEPS):
        response = await llm.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, response_format=REPORT_FORMAT,
        )
        trace["llm_calls"] += 1
        trace["input_tokens"] += response.usage.prompt_tokens
        trace["output_tokens"] += response.usage.completion_tokens
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            report = json.loads(msg.content)
            # 승인 대기열에 넣기 전에 검증만 먼저 해 본다. 거부된 제안이 있으면 이유를 알려 주고 한 번 고쳐 쓰게 한다
            # (전에는 거부된 제안이 조용히 버려져서, 방향은 맞았던 OOM 수정이 아예 사람에게 가지 못했다)
            errors = validation_errors(report)
            if errors and len(trace["revisions"]) < MAX_REVISIONS:
                trace["revisions"].append(errors)
                messages.append({"role": "user", "content":
                    "보고서의 다음 제안이 검증에서 거부됐다. 허용 범위 안에서 고쳐서 보고서 전체를 다시 써라. "
                    "(필요하면 도구를 다시 써도 된다)\n" + "\n".join(f"- {j}: {e}" for j, e in errors.items())})
                continue
            trace["latency_s"] = round(time.perf_counter() - t0, 1)
            return {"report": report, "trace": trace, "requests": file_requests(report)}

        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            try:
                r = await mcp.call_tool(call.function.name, args)
                result = r.content[0].text
            except Exception as e:
                result = json.dumps({"error": f"도구 실행 오류: {e}"}, ensure_ascii=False)
            trace["tool_calls"].append({"name": call.function.name, "args": args})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    trace["latency_s"] = round(time.perf_counter() - t0, 1)
    return {"report": {"jobs": []}, "trace": trace, "error": "MAX_STEPS 초과"}


# 근거 대조에 쓰는 "고유한 증거" 토큰: 예외 이름(KeyError, OutOfMemoryError...)과 shard 파일 이름
EVIDENCE_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z]*Error\b|shard-\d{4}")


def grounding_error(job_id: str, evidence: str):
    """보고서 근거에 인용한 예외 이름·shard 이름이 그 job의 로그에 실제로 있는지 확인한다.
    없으면 다른 job의 진단을 베꼈을 가능성 (troubleshooting_log 16번). 이유 문자열을 돌려주고, 문제없으면 None."""
    log = "\n".join(jobs.tail(job_id, 100000))
    missing = sorted({t for t in EVIDENCE_TOKEN_RE.findall(evidence) if t not in log})
    if missing:
        return (f"근거에 인용한 {missing}이(가) 이 job({job_id})의 로그에 없다. 다른 job과 혼동했는지 "
                f"tail_log로 이 job의 로그를 다시 확인하라")
    return None


def validation_errors(report: dict) -> dict:
    """대기열에 넣지 않고 검증만. {job_id: 거부 이유}
    1) 근거 대조: 인용한 오류가 그 job 로그에 실제로 있나  2) 제안 검증: 허용 목록·범위·체크포인트 상태"""
    errors = {}
    for j in report["jobs"]:
        try:
            jobs.job_dir(j["job_id"])
            if j["action_requested"] != "none" and (g := grounding_error(j["job_id"], j.get("evidence", ""))):
                raise ValueError(g)
            if j["action_requested"] == "restart":
                changes = {k: v for k, v in (j.get("changes") or {}).items() if v is not None}
                jobs.validate_changes(j["job_id"], changes, j.get("resume_from_step"))
                # 메모리 기반 검사: 거의 같은 장애에서 "이 종류의 조치"가 여러 번 실패하고 한 번도 성공 못 했으면 거부
                # (프롬프트로 "실패한 방법은 반복하지 마라"고 했지만, 정책 문장과 충돌하면 정책을 따랐다 → 21번)
                kinds = sorted(changes) or ["(변경 없음)"]
                for rb in runbooks.find(j["job_id"]):
                    if rb["similarity"] >= 0.8 and rb["change_kinds"] == kinds and rb["failures"] >= 2 \
                            and rb["verified_successes"] == 0:
                        others = [b for b in runbooks.find(j["job_id"], k=5) if b["verified_successes"] > 0]
                        hint = (f" 성공 기록이 있는 방법: {[(b['change_kinds'], b['recent'][-1]['lesson']) for b in others]}"
                                if others else "")
                        raise ValueError(f"거의 같은 장애(유사도 {rb['similarity']})에서 {kinds} 조치가 "
                                         f"{rb['failures']}번 실패하고 한 번도 성공하지 못했다 (runbook {rb['runbook_id']})."
                                         f"{hint}")
        except ValueError as e:
            if j["action_requested"] != "none":
                errors[j["job_id"]] = str(e)
    return errors


def file_requests(report: dict) -> list[dict]:
    """보고서에 적힌 조치를 코드가 승인 대기열에 넣는다 → 보고서(말)와 대기열(행동)이 항상 일치한다."""
    filed = []
    for j in report["jobs"]:
        if j["action_requested"] == "none":
            continue
        try:
            if (g := grounding_error(j["job_id"], j.get("evidence", ""))):  # 고쳐 쓴 뒤에도 근거가 안 맞으면 요청하지 않는다
                raise ValueError(g)
            extra = {}
            if j["action_requested"] == "restart":
                extra = {"changes": {k: v for k, v in (j.get("changes") or {}).items() if v is not None},
                         "resume_from": j.get("resume_from_step")}
            item = approvals.request(j["job_id"], j["action_requested"], f"[{j['diagnosis']}] {j['evidence']}", **extra)
            filed.append({"job_id": j["job_id"], "action": item["action"], "approval_id": item["id"],
                          "changes": item.get("changes"), "resume_from": item.get("resume_from")})
        except ValueError as e:  # LLM이 없는 job_id를 지어낸 경우, 허용 범위 밖의 설정을 제안한 경우 등
            filed.append({"job_id": j["job_id"], "action": j["action_requested"], "error": str(e)})
    return filed


async def run_once() -> dict:
    # 순찰 전에, 지난번에 실행된 조치들의 결과를 확인해서 runbook에 기록한다 (성공이 확인된 것만 "검증된 조치"가 된다)
    approvals.verify_outcomes()
    async with Client(ops_server()) as mcp:
        listed = await mcp.list_tools()
        tools = [{"type": "function",
                  "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
                 for t in listed.tools]
        return await patrol(AsyncOpenAI(), mcp, tools)


def print_report(result: dict) -> None:
    for j in result["report"]["jobs"]:
        act = "" if j["action_requested"] == "none" else f"  → 승인 요청: {j['action_requested']}"
        print(f"[{j['job_id']}] {j['diagnosis']}{act}")
        print(f"    근거: {j['evidence']}")
        print(f"    권고: {j['recommendation']}")
    t = result["trace"]
    calls = ", ".join(f"{c['name']}({c['args'].get('job_id', '')})" for c in t["tool_calls"])
    print(f"\nLLM 호출 {t['llm_calls']}회, 도구 호출 {len(t['tool_calls'])}회, {t['latency_s']}s, "
          f"토큰 in {t['input_tokens']} / out {t['output_tokens']}")
    print(f"도구: {calls}")
    for r in result.get("requests", []):
        print(f"승인 요청: {r}")
    if result.get("error"):
        print("!", result["error"])


if __name__ == "__main__":
    print_report(asyncio.run(run_once()))
