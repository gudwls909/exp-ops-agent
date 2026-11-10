# 조회·진단·승인 API 이미지 (Agent + MCP 서버 + FastAPI)
# 학습 job은 호스트에서 돈다. 이 컨테이너는 호스트의 job을 "보고" "승인된 조치를 실행"한다 → 실행할 때 옵션이 중요하다:
#   --pid=host            : 호스트 프로세스(학습 job)의 생존 확인·종료를 위해 필요
#   --user $(id -u):$(id -g) : root로 돌리지 않는다. 같은 사용자의 job만 종료할 수 있게 (root면 호스트의 아무 프로세스나 죽일 수 있다)
#   --security-opt apparmor=unconfined : Docker 기본 AppArmor 정책은 컨테이너 밖 프로세스에 신호를 못 보내게 막는다.
#                         같은 사용자라도 종료(SIGTERM)가 PermissionError로 실패했다 (troubleshooting_log 25번).
#                         정책을 끄는 대신 필요한 신호만 허용하는 전용 AppArmor 프로필을 만드는 게 더 안전하다
#   -v $(pwd)/runs:/app/runs -v $(pwd)/state:/app/state : job 로그·체크포인트, 승인 대기열·runbook
# ⚠️ 재시작(restart)은 학습 환경(torch, GPU)이 필요해서 이 이미지 안에서는 실행되지 않는다 (README 참고)
FROM python:3.11-slim

WORKDIR /app
# 의존성 먼저 → 코드만 바뀌면 이 층은 캐시를 재사용한다 (week1에서 배운 것)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ops ./ops
COPY agent ./agent
COPY api ./api

EXPOSE 8100
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8100"]
