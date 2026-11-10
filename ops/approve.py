"""
사람이 쓰는 승인 CLI (나중에 FastAPI + 웹/슬랙 버튼으로 바뀔 부분)

  python ops/approve.py            # 대기 중인 요청 보기
  python ops/approve.py 3 --yes    # 3번 요청 승인 → 실제 실행
  python ops/approve.py 3 --no     # 3번 요청 거절
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import approvals  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("id", nargs="?", type=int)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--yes", action="store_true")
    g.add_argument("--no", action="store_true")
    args = p.parse_args()

    if args.id is None:
        items = approvals.pending()
        if not items:
            print("대기 중인 요청 없음")
        for it in items:
            print(f"[{it['id']}] {it['action']:<9} {it['job_id']:<12} {it['created_at']}  이유: {it['reason']}")
        return

    if not (args.yes or args.no):
        sys.exit("--yes 또는 --no를 붙여야 합니다")
    item = approvals.decide(args.id, approve=args.yes)
    print(f"[{item['id']}] {item['status']}: {item.get('result', '')}")


if __name__ == "__main__":
    main()
