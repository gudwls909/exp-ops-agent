"""
memory ablation 결과 여러 개(seed별 파일)를 합쳐서 통계로 본다

왜 짝지어(paired) 비교하나
- 같은 seed, 같은 에피소드 번호면 memory off/on이 "같은 장애"(같은 shard 번호, 같은 batch)를 본다
- 그래서 (off 라운드 수 − on 라운드 수)를 짝마다 계산하면 장애 난이도 차이가 빠지고 memory 효과만 남는다
- 해결 못 함은 4라운드로 친다 (최대 3라운드 + 벌점 1)

신뢰구간: 짝 차이를 bootstrap(복원 추출 10000번)해서 평균 차이의 95% 구간.
  구간이 0을 포함하면 "효과가 있다"고 말할 수 없다.

실행:
  python eval/summarize_memory.py eval/results/*_memory_s1.json eval/results/*_memory_s2.json
"""

import json
import random
import sys
from collections import defaultdict

FAIL = 4


def rounds(x):
    return x["resolved_round"] or FAIL


def main(paths):
    rows = []  # (seed, cond, episode, job, rounds, cost)
    for p in paths:
        seed = p.split("_memory_s")[-1].split(".")[0]
        for r in json.load(open(p)):
            for job, x in r["jobs"].items():
                rows.append((seed, r["condition"], r["episode"], job, rounds(x), r["cost_usd"] / len(r["jobs"])))
            rows.append((seed, r["condition"], r["episode"], "_touched", int(r["touched_healthy"]), 0))

    jobs = sorted({r[3] for r in rows if not r[3].startswith("_")})
    print(f"파일 {len(paths)}개, seed {sorted({r[0] for r in rows})}\n")

    for job in jobs + ["전체"]:
        sel = [r for r in rows if (r[3] == job if job != "전체" else not r[3].startswith("_"))]
        print(f"=== {job} ===")
        for cond in ("memory_off", "memory_on"):
            c = [r for r in sel if r[1] == cond]
            solved = sum(r[4] < FAIL for r in c)
            first = sum(r[4] == 1 for r in c)
            later = [r for r in c if r[2] >= 2]  # 에피소드 2부터: memory on은 이전 기록을 가질 수 있음
            first_later = sum(r[4] == 1 for r in later)
            print(f"  {cond:<11} 해결 {solved}/{len(c)}  1라운드 해결 {first}/{len(c)}  "
                  f"(ep2 이후 {first_later}/{len(later)})  평균 라운드(실패=4) {sum(r[4] for r in c) / len(c):.2f}")
        # 짝 비교
        pairs = defaultdict(dict)
        for s, cond, ep, j, rd, _ in sel:
            pairs[(s, ep, j)][cond] = rd
        diffs = [v["memory_off"] - v["memory_on"] for v in pairs.values() if len(v) == 2]
        if diffs:
            better = sum(d > 0 for d in diffs)
            worse = sum(d < 0 for d in diffs)
            rng = random.Random(0)
            boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(10000))
            lo, hi = boots[250], boots[9749]
            print(f"  짝 {len(diffs)}개: on이 빠름 {better}, 같음 {len(diffs) - better - worse}, on이 느림 {worse}  "
                  f"평균 차이(off−on) {sum(diffs) / len(diffs):+.2f} 라운드, 95% CI [{lo:+.2f}, {hi:+.2f}]"
                  f"{'  ← 0 포함: 효과 주장 불가' if lo <= 0 <= hi else ''}")
        print()

    # 학습 곡선: 에피소드 번호별 평균 라운드
    print("=== 에피소드별 평균 라운드 (전체 job, 실패=4) ===")
    eps = sorted({r[2] for r in rows})
    for cond in ("memory_off", "memory_on"):
        vals = []
        for ep in eps:
            c = [r[4] for r in rows if r[1] == cond and r[2] == ep and not r[3].startswith("_")]
            vals.append(f"{sum(c) / len(c):.2f}" if c else "-")
        print(f"  {cond:<11} " + "  ".join(f"ep{e}:{v}" for e, v in zip(eps, vals)))

    print("\n=== 비용·안전 ===")
    for cond in ("memory_off", "memory_on"):
        eps_cost = defaultdict(float)
        for s, c, ep, j, _, cost in rows:
            if c == cond:
                eps_cost[(s, ep)] += cost
        touched = sum(r[4] for r in rows if r[1] == cond and r[3] == "_touched")
        print(f"  {cond:<11} 에피소드당 비용 ${sum(eps_cost.values()) / len(eps_cost):.4f}, 정상 job 건드림 {touched}/{len(eps_cost)}")


if __name__ == "__main__":
    main(sys.argv[1:])
