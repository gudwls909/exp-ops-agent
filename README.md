# Experiment Ops Agent

GPU 학습 job의 장애(hang, OOM, NaN, crash)를 감지하고 원인을 요약한 뒤, **사람의 승인을 받아** 종료·재시작하는 Agent.

> 진행 중인 프로젝트입니다.

## 구성 (진행 상황)

- [x] `sim/fake_train.py`: 실패를 원하는 시점에 주입할 수 있는 실제 GPU 학습 job (평가용)
- [x] `baseline/rule_watchdog.py`: 규칙 기반 감시기 (로그 정지 시간 + 프로세스 생존). 비교 기준
- [x] MCP ops 서버 (`ops/`): job 조회 도구 4개 + 승인 요청 도구. 실행(kill/restart) 도구는 Agent에게 주지 않고 사람 승인 후에만 실행
- [x] 안전장치 테스트 (`tests/test_safety.py`): 실행 도구 미노출, 승인 전 미실행, job_id 검증, pid 재사용 방어, SIGTERM→SIGKILL, 정답 누출 방지
- [ ] Agent + 승인 흐름
- [ ] 평가: 실패 유형별 감지율, 진단 정확도, 잘못 죽인 비율(false kill), 감지 시간, 비용
