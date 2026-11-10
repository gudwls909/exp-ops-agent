"""
MCP ops 서버 — week1 step3_docs_server.py와 같은 구조, 도구만 다르다

Agent에게 주는 도구: 읽기 전용 6개 (list_jobs, job_status, tail_log, gpu_status, job_config, list_checkpoints)
  - 실행 도구(kill/restart)는 아예 없다
  - 승인 요청도 도구로 주지 않는다. Agent가 보고서에 쓴 조치를 "코드"가 승인 대기열에 넣는다 (agent/agent.py)
    이유: 도구로 줬더니 보고서엔 "terminate 요청"이라 쓰고 실제로는 도구를 안 부른 경우가 있었다 (말과 행동 불일치)

도구 결과는 JSON 문자열로 돌려준다 (LLM이 읽기 좋고, 평가 코드가 파싱하기도 좋다)
stdio 서버이므로 print()는 금지. 로그는 stderr로 (week1에서 배운 것)
"""

import json
import os
import sys

from mcp.server.mcpserver import MCPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jobs  # noqa: E402

mcp = MCPServer("ops")
MAX_TAIL = 200


def _ok(data) -> str:
    return json.dumps(data, ensure_ascii=False)


def _err(e: Exception) -> str:
    # 잘못된 입력은 예외로 서버를 죽이지 않고, 이유를 LLM에게 돌려준다 (week1 step2의 도구 오류 처리와 같은 원칙)
    return _ok({"error": str(e)})


@mcp.tool()
def list_jobs() -> str:
    """모든 학습 job의 현재 상태 요약 목록. 먼저 이걸로 전체를 훑어본다.
    각 항목: job_id, alive(프로세스 생존), log_age_s(로그가 마지막으로 바뀐 뒤 지난 초), last_step/total_steps, gpu_mem_mib"""
    mem = jobs.gpu_mem_by_pid()  # nvidia-smi를 job마다 부르지 않고 한 번만
    return _ok([jobs.summary(j, mem) for j in jobs.list_job_ids()])


@mcp.tool()
def job_status(job_id: str) -> str:
    """job 하나의 상태 요약 (list_jobs의 한 항목과 같은 내용)."""
    try:
        return _ok(jobs.summary(job_id))
    except ValueError as e:
        return _err(e)


@mcp.tool()
def tail_log(job_id: str, n: int = 30) -> str:
    """job의 학습 로그(train.log) 마지막 n줄. 최대 200줄. 원인 진단에 쓴다 (traceback, loss 값, 마지막 메시지 등)."""
    try:
        n = max(1, min(int(n), MAX_TAIL))
        return _ok({"job_id": job_id, "lines": jobs.tail(job_id, n)})
    except ValueError as e:
        return _err(e)


@mcp.tool()
def gpu_status() -> str:
    """GPU별 사용률(%)과 메모리 사용량. 그리고 GPU 메모리를 쓰고 있는 job (pid → job_id)."""
    mem = jobs.gpu_mem_by_pid()
    pid_to_job = {}
    for j in jobs.list_job_ids():
        pid = jobs.read_job(j)["pid"]
        if pid in mem and jobs.is_alive(pid, j):
            pid_to_job[pid] = j
    return _ok({
        "gpus": jobs.gpus(),
        "processes": [{"pid": p, "mem_mib": m, "job_id": pid_to_job.get(p)} for p, m in mem.items()],
    })


@mcp.tool()
def job_config(job_id: str) -> str:
    """job의 학습 설정(하이퍼파라미터): max_lr, warmup_steps, batch_size, stage2_at/stage2_batch/stage2_lr(커리큘럼 2단계),
    grad_accum, data_shards 등. 실패 원인이 설정에 있는지 볼 때, 고칠 설정을 제안할 때 쓴다."""
    try:
        return _ok({"job_id": job_id, "config": jobs.read_config(job_id)})
    except ValueError as e:
        return _err(e)


@mcp.tool()
def list_checkpoints(job_id: str) -> str:
    """job의 step별 체크포인트 목록. 각 항목: step, loss_logged_at_step, loss_at_next_log.
    체크포인트 N은 step N의 업데이트가 끝난 뒤의 가중치다. step N 로그의 loss는 업데이트 '전' 값이라,
    그 가중치가 멀쩡한지는 loss_at_next_log(그 가중치에서 출발한 다음 로그의 loss)로 판단하라.
    loss가 망가지기(nan/폭주) '전'의 체크포인트를 골라 재시작을 제안할 때 쓴다."""
    try:
        return _ok({"job_id": job_id, "checkpoints": jobs.checkpoints(job_id)})
    except ValueError as e:
        return _err(e)


if __name__ == "__main__":
    print("ops MCP 서버 시작 (stdio)", file=sys.stderr)
    mcp.run()
