"""
Ops Agent — MCP ops 서버에 붙어서 job들을 한 번 "순찰"하고, 진단 보고서와 승인 요청을 남긴다

week1 step3_agent.py / step5_api.py의 run_agent와 같은 루프다. 달라진 점
1. 시스템 프롬프트에 "운영 정책"이 들어간다 (언제 종료를 요청하고, 언제 건드리지 말아야 하나)
2. 마지막 답을 자유 텍스트가 아니라 JSON 보고서로 받는다 (Structured Outputs)
   → 사람이 읽기도 좋고, 평가 코드가 "진단이 맞았나"를 자동 채점할 수 있다 (week1 judge.py와 같은 기법)
3. 한 번의 순찰에 도구를 여러 번 부른다 (job 여러 개 × 상태 확인/로그 확인) → MAX_STEPS를 넉넉히

실행:
  python agent/agent.py            # 순찰 1회 → 보고서 출력. 승인 요청은 python ops/approve.py 로 확인
"""

import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from openai import AsyncOpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
MAX_STEPS = 15
OPS_SERVER = StdioServerParameters(command=sys.executable, args=[os.path.join(ROOT, "ops", "server.py")],
                                   env={**os.environ})  # OPS_RUNS_DIR 같은 설정을 서버에도 넘긴다 (평가 때 사용)

DIAGNOSES = ["healthy", "slow_but_progressing", "finished", "hung", "hung_at_shutdown",
             "crashed", "oom", "nan_loss", "unknown"]

SYSTEM_PROMPT = """너는 GPU 학습 서버를 돌보는 운영(ops) 엔지니어다. 모든 학습 job을 점검하고 진단 보고서를 쓴다.

점검 절차
1. list_jobs로 전체 상태를 본다.
2. 조금이라도 이상해 보이는 job(프로세스가 죽었음, 로그가 오래 안 바뀜, step이 total에 못 미친 채 멈춤 등)은
   반드시 tail_log로 로그를 직접 확인한 뒤에 진단한다. 숫자 요약만 보고 진단하지 마라.
3. 필요하면 gpu_status, job_status를 쓴다.

진단 종류 (diagnosis)
- healthy              : 정상 진행 중
- slow_but_progressing : 느려졌지만 step이 늘고 loss가 정상 값으로 계속 찍힌다 → 정상. 절대 종료 요청하지 마라
- finished             : 정상 종료 (Training finished 후 프로세스 종료)
- hung                 : 학습 도중 멈춤 (프로세스는 살아 있는데 로그가 진행되지 않음)
- hung_at_shutdown     : 학습은 끝났다는 로그(Iteration limit reached 등)가 있는데 프로세스가 안 끝남
- crashed              : 예외(traceback)로 죽음 (OOM 제외)
- oom                  : CUDA out of memory로 죽음
- nan_loss             : loss가 nan/inf인데 계속 돌고 있음 (GPU 낭비)
- unknown              : 판단할 근거가 부족함

조치 정책 (request_action)
- hung, hung_at_shutdown, nan_loss → terminate 요청
- hung으로 종료된 job은 이전 체크포인트가 정상이므로 종료 후 restart가 적절하다. 단 아직 살아 있으면 terminate만 먼저 요청한다
- crashed, oom → 이미 죽었으므로 요청하지 않는다. 그대로 재시작하면 같은 실패가 반복되므로 원인과 고칠 점을 보고서에 쓴다
- healthy, slow_but_progressing, finished, unknown → 요청하지 않는다
- 실험을 잘못 죽이는 것이 가장 비싼 실수다. 확신이 없으면 요청하지 말고 unknown으로 보고하라
- request_action의 reason에는 사람이 바로 판단할 수 있게 근거(로그 내용, 수치)를 쓴다

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
                            "recommendation": {"type": "string"},
                        },
                        "required": ["job_id", "diagnosis", "evidence", "action_requested", "recommendation"],
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
    trace = {"tool_calls": [], "llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
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
            trace["latency_s"] = round(time.perf_counter() - t0, 1)
            return {"report": json.loads(msg.content), "trace": trace}

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


async def run_once() -> dict:
    async with Client(OPS_SERVER) as mcp:
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
    if result.get("error"):
        print("!", result["error"])


if __name__ == "__main__":
    print_report(asyncio.run(run_once()))
