# Experiment Ops Agent

GPU 학습 job의 장애(hang, OOM, NaN, crash)를 감지하고 원인을 요약한 뒤, **사람의 승인을 받아** 종료·재시작하는 Agent.

> 진행 중인 프로젝트입니다.

## 구성 (진행 상황)

- [x] `sim/fake_train.py`: 실패를 원하는 시점에 주입할 수 있는 실제 GPU 학습 job (평가용)
- [x] `baseline/rule_watchdog.py`: 규칙 기반 감시기 (로그 정지 시간 + 프로세스 생존). 비교 기준
- [ ] MCP ops 서버: job 조회 도구 + 승인 요청 도구
- [ ] Agent + 승인 흐름
- [ ] 평가: 실패 유형별 감지율, 진단 정확도, 잘못 죽인 비율(false kill), 감지 시간, 비용
