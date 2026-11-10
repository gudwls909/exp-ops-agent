"""
규칙 기반 감시기 (베이스라인) — vb_hang_killer.sh를 Python으로 일반화한 것

판정 규칙 (vb_hang_killer.sh와 같은 아이디어)
  프로세스가 죽었다                        → EXITED
  프로세스가 살아 있고, 로그가 N초 넘게 안 바뀌었다 → HANG
  그 외                                   → RUNNING

⚠️ 판정만 출력한다. 실제로 kill하지 않는다. (dry-run)

실행:
  python baseline/rule_watchdog.py --threshold 15

채울 곳은 TODO 1~4. 위에서부터 하나씩 채우고, 하나 채울 때마다 실행해서 확인하자.
"""

import argparse
import glob
import json
import os
import time

RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def find_jobs():
    """runs/ 아래의 job 폴더 경로 목록을 돌려준다. 예: ['.../runs/h1', '.../runs/s1']"""
    # TODO 1: glob.glob으로 runs/ 안의 폴더들을 찾아서 정렬한 리스트를 return
    #   힌트: glob.glob(os.path.join(RUNS_DIR, "*"))  → 경로 문자열 리스트
    #         sorted(...)로 정렬하면 출력 순서가 매번 같다
    return sorted(glob.glob(os.path.join(RUNS_DIR, "*")))


def read_pid(job_dir):
    """job.json에서 pid(프로세스 번호)를 읽어 돌려준다."""
    # TODO 2: job_dir 안의 job.json을 열어서 json.load로 dict를 만들고 "pid" 값을 return
    #   힌트: week1 db.py의 json.loads와 같은 개념. 파일이면 json.load(f)
    #         with open(os.path.join(job_dir, "job.json")) as f:
    with open(os.path.join(job_dir, "job.json")) as f:
        job_info = json.load(f)
    return job_info.get("pid", None)


def is_alive(pid):
    """그 pid의 프로세스가 살아 있으면 True."""
    # TODO 3: os.kill(pid, 0)을 try로 감싸기
    #   os.kill(pid, 0)은 "신호 0번" = 아무것도 안 보내고 "존재하는지만" 확인한다 (프로세스를 죽이지 않음!)
    #   프로세스가 없으면 ProcessLookupError 예외가 난다 → 그때 False
    #   (vb_hang_killer.sh의 ps | grep 부분을 대신하는 것)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def log_age(job_dir):
    """train.log가 마지막으로 수정된 지 몇 초 지났는지."""
    # TODO 4: 지금 시각 - 파일 수정 시각
    #   힌트: time.time()                   → 지금 (초)
    #         os.path.getmtime(파일 경로)    → 마지막 수정 시각 (초)
    #   (vb_hang_killer.sh의 age=$(( $(date +%s) - $(stat -c %Y "$pl") )) 와 똑같다)
    log_path = os.path.join(job_dir, "train.log")
    if not os.path.exists(log_path):
        return float("inf")
    return time.time() - os.path.getmtime(log_path)


def judge(job_dir, threshold):
    """위의 함수들을 조합해서 판정 하나를 돌려준다. (이건 완성돼 있다)"""
    pid = read_pid(job_dir)
    if pid is None or not is_alive(pid):
        return "EXITED"
    if log_age(job_dir) > threshold:
        return "HANG"
    return "RUNNING"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=120, help="로그가 이 초 이상 안 바뀌면 HANG")
    args = parser.parse_args()

    jobs = find_jobs()
    if not jobs:
        print("job이 없습니다 (runs/ 폴더가 비었거나 TODO 1이 아직 안 됨)")
    for job_dir in jobs:
        name = os.path.basename(job_dir)
        print(f"{name:<12} {judge(job_dir, args.threshold):<8} log_age={log_age(job_dir):.0f}s pid={read_pid(job_dir)}")


if __name__ == "__main__":
    main()
