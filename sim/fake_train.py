"""
가짜 학습 job: 진짜 GPU에서 작은 모델을 학습시키면서, 원하는 순간에 원하는 실패를 일으킨다

왜 필요한가
- 진짜 실험(AutoSkill 등)은 실패가 "언제 날지" 모른다 → Agent 개발/평가를 반복할 수 없다
- 이 스크립트는 --fail 옵션 하나로 실패 종류와 시점을 고를 수 있다 → 평가 케이스를 원하는 만큼 만든다
- 학습 자체는 진짜다 (GPU 메모리, 사용률, 프로세스, 로그가 실제로 생긴다) → Agent가 보는 신호도 진짜

실패 종류 (--fail)
  none              정상 종료
  hang              학습 도중 멈춤. 프로세스는 살아 있고 GPU 메모리도 잡고 있지만 로그가 더 안 나온다 (데드락 흉내)
  hang_at_shutdown  학습은 다 끝났는데 종료하다가 멈춤 ← vb_hang_killer.sh가 처리하던 바로 그 상황
  oom               실제로 GPU 메모리를 넘치게 할당해서 CUDA OOM으로 죽는다
  nan               학습률이 폭주해서 loss가 nan이 된다. 죽지 않고 nan을 찍으며 GPU를 계속 낭비한다
  crash             데이터 오류로 예외가 나서 죽는다 (traceback이 로그에 남음)
  slow              ⚠️ 함정: 느려질 뿐 정상이다. 로그 간격이 길어져서 "로그가 N초 안 바뀌면 kill" 규칙에 걸린다
                    → 죽이면 안 되는 job. Agent의 "잘못 죽임(false kill)"을 재기 위한 케이스

예시
  python sim/fake_train.py --job-id demo1 --fail none
  python sim/fake_train.py --job-id demo2 --fail hang --fail-at 50
  python sim/fake_train.py --job-id demo2 --resume          # 마지막 체크포인트부터 이어서 (재시작 테스트용)

주의
- oom은 실제로 그 GPU의 빈 메모리를 전부 잡는다. 다른 사람이 같은 GPU를 쓰고 있으면 그 job도 죽을 수 있다.
  비어 있는 GPU에서만 쓰고 --gpu로 GPU를 지정하라.
- 실패 종류(정답)는 job 폴더에 기록하지 않는다. Agent가 정답을 훔쳐보면 평가가 무의미해지므로,
  정답은 나중에 평가 실행기가 따로 보관한다.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time

import torch
import torch.nn as nn

FAILS = ["none", "hang", "hang_at_shutdown", "oom", "nan", "crash", "slow"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", required=True)
    p.add_argument("--fail", choices=FAILS, default="none")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--fail-at", type=int, default=None, help="몇 step에서 실패할지 (기본: 전체의 절반)")
    p.add_argument("--step-time", type=float, default=0.1, help="step당 최소 시간(초). 실험을 사람이 볼 만한 속도로")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--slow-factor", type=float, default=20, help="slow 모드에서 몇 배 느려지나")
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--runs-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"))
    p.add_argument("--resume", action="store_true", help="마지막 체크포인트부터 이어서 학습")
    p.add_argument("--resume-from", type=int, default=None, help="이 step의 체크포인트부터 이어서 학습")
    p.add_argument("--seed", type=int, default=0)

    # ---- 학습 설정 (config.json에 기록되고 Agent가 볼 수 있다. 실패가 "설정 때문에" 자연스럽게 일어난다) ----
    p.add_argument("--max-lr", type=float, default=1e-3, help="warmup이 끝난 뒤의 학습률. 너무 크면 loss가 발산(nan)")
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--stage2-at", type=int, default=-1, help="이 step부터 커리큘럼 2단계 (-1이면 없음)")
    p.add_argument("--stage2-batch", type=int, default=1024, help="2단계 batch size. 너무 크면 CUDA OOM")
    p.add_argument("--stage2-lr", type=float, default=-1, help="2단계 학습률 (-1이면 그대로). 너무 크면 loss 발산")
    p.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation. batch를 이만큼 나눠 계산 → 메모리 절약, 실질 batch는 같음")
    p.add_argument("--data-shards", default="0,1,2,3,4,5,6,7", help="사용할 데이터 shard 번호 목록")
    p.add_argument("--shard-steps", type=int, default=40, help="shard 하나를 몇 step 동안 읽나")

    # ---- 시뮬레이터 환경 (Agent에게 보이지 않는다. "디스크에 있는 손상된 파일" 같은 바깥 상황) ----
    p.add_argument("--bad-shards", default="", help="손상된 shard 번호 (예: 3). 이 shard를 읽으면 KeyError")
    return p.parse_args()


# Agent가 볼 수 있는 학습 설정만 골라서 config.json에 쓴다 (실패 주입 옵션은 빼고)
CONFIG_KEYS = ["steps", "max_lr", "warmup_steps", "batch_size", "stage2_at", "stage2_batch", "stage2_lr", "grad_accum",
               "data_shards", "shard_steps", "ckpt_every", "log_every"]


def lr_at(step, args):
    """선형 warmup 후 max_lr 유지. 커리큘럼 2단계에서 stage2_lr이 있으면 그 값으로 바뀐다"""
    if args.stage2_at >= 0 and step >= args.stage2_at and args.stage2_lr > 0:
        return args.stage2_lr
    return args.max_lr * min(1.0, step / max(1, args.warmup_steps))


def ckpt_file(job_dir, step):
    return os.path.join(job_dir, f"ckpt-{step:06d}.pt")


def setup_logging(log_path):
    # 실제 학습 코드처럼 시간 + 레벨 + 메시지. 파일과 터미널 둘 다에 쓴다.
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def load_batch(step, batch_size, device, true_w, corrupt_at, shard=None, bad_shards=()):
    """데이터 로더 흉내. 손상된 shard를 읽거나(corrupt_at step이면) 손상된 샘플을 만난 것처럼 예외를 낸다."""
    if step == corrupt_at:
        raise KeyError(f"'label' missing in sample index {step * 32 + 7} (corrupted shard data/shard-0003.tar)")
    if shard is not None and shard in bad_shards:
        raise KeyError(f"'label' missing in sample index {step * 32 + 7} (corrupted shard data/shard-{shard:04d}.tar)")
    x = torch.randn(batch_size, 512, device=device)
    y = x @ true_w + 0.1 * torch.randn(batch_size, 1, device=device)
    return x, y


def hang_forever(log=None):
    # 데드락 흉내: 아무것도 안 하고 영원히 잠든다. 프로세스는 살아 있고, 잡은 GPU 메모리도 그대로다.
    while True:
        time.sleep(3600)


def main():
    args = parse_args()
    fail_at = args.fail_at if args.fail_at is not None else args.steps // 2
    job_dir = os.path.join(args.runs_dir, args.job_id)
    ckpt_path = os.path.join(job_dir, "ckpt.pt")
    os.makedirs(job_dir, exist_ok=True)
    log = setup_logging(os.path.join(job_dir, "train.log"))

    # job.json: 스케줄러가 남기는 기록 같은 것. "어떻게 다시 실행하나"를 알아야 재시작할 수 있다.
    # (실패 종류는 일부러 적지 않는다 → 위의 "주의" 참고. 다만 재실행 명령에는 --fail이 들어 있으니
    #  나중에 Agent에게 보여 줄 때는 명령줄을 가려야 한다 → 과제)
    with open(os.path.join(job_dir, "job.json"), "w") as f:
        json.dump({"job_id": args.job_id, "pid": os.getpid(), "python": sys.executable, "cmd": sys.argv,
                   "cwd": os.getcwd(), "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    # config.json: 실제 학습 코드가 남기는 hparams 기록 같은 것. Agent는 이걸 보고 고칠 설정을 제안한다
    with open(os.path.join(job_dir, "config.json"), "w") as f:
        json.dump({k: getattr(args, k) for k in CONFIG_KEYS}, f, indent=2)
    shards = [int(x) for x in args.data_shards.split(",") if x.strip() != ""]
    bad_shards = {int(x) for x in args.bad_shards.split(",") if x.strip() != ""}

    # SIGTERM(일반 kill)을 받으면 체크포인트를 저장하고 깔끔하게 끝낸다.
    # kill -9(SIGKILL)는 잡을 수 없다 → 체크포인트 없이 즉사. Agent가 어떤 kill을 쓰느냐가 중요한 이유.
    state = {"step": 0}

    def on_sigterm(signum, frame):
        log.warning(f"Received SIGTERM at step {state['step']}, saving checkpoint and exiting")
        ck = {"step": state["step"], "model": model.state_dict()}
        torch.save(ck, ckpt_path)
        torch.save(ck, ckpt_file(job_dir, state["step"]))
        sys.exit(143)

    signal.signal(signal.SIGTERM, on_sigterm)

    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # 작은 MLP + 가짜 회귀 데이터. 목적은 "진짜 GPU 연산이 돈다"는 것뿐이다.
    model = nn.Sequential(nn.Linear(512, 2048), nn.ReLU(), nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    true_w = torch.randn(512, 1, device=device) / 512 ** 0.5  # 정답 y의 분산이 1 근처가 되게 (loss가 보기 좋게 줄어든다)
    batch_size = 256

    start = 0
    resume_path = ckpt_file(job_dir, args.resume_from) if args.resume_from is not None else ckpt_path
    if (args.resume or args.resume_from is not None) and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck["model"])
        start = ck["step"]
        log.info(f"Resumed from checkpoint at step {start}")
        if args.fail != "none" and fail_at <= start:
            fail_at = args.steps + 1  # 이미 지난 실패 지점은 다시 일으키지 않는다 (재시작 성공 시나리오)

    log.info(f"Start training job={args.job_id} device={device} steps={args.steps} pid={os.getpid()}")

    for step in range(start + 1, args.steps + 1):
        state["step"] = step
        t0 = time.time()

        # ---------------- 실패 주입 ----------------
        if step == fail_at:
            if args.fail == "hang":
                hang_forever()
            if args.fail == "oom":
                # 실제로 흔한 원인: 커리큘럼/설정 실수로 batch size가 갑자기 커짐 → forward에서 진짜 CUDA OOM
                batch_size = 4_194_304
                log.info(f"curriculum: entering stage 2, batch_size={batch_size}")
            if args.fail == "nan":
                for g in opt.param_groups:
                    g["lr"] = 1e12  # 학습률 폭주 → 몇 step 안에 inf → nan
                log.info("lr scheduler: warmup finished, lr=1.0e+12")  # 실제로 흔한 원인: 스케줄러 설정 실수
        slow = args.fail == "slow" and step >= fail_at

        # 설정 기반 동작: 학습률 스케줄, 커리큘럼 2단계 batch, 데이터 shard
        if args.fail != "nan":  # (예전 방식 --fail nan은 위에서 lr을 직접 바꾼다)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args)
        if step == args.stage2_at:
            batch_size = args.stage2_batch
            log.info(f"curriculum: entering stage 2, batch_size={batch_size} grad_accum={args.grad_accum} "
                     f"lr={lr_at(step, args):.1e}")
        shard = shards[(step - 1) // args.shard_steps % len(shards)] if shards else None
        if shard is not None and (step - 1) % args.shard_steps == 0:
            log.info(f"data: reading shard data/shard-{shard:04d}.tar")

        # gradient accumulation: batch를 grad_accum개로 나눠 forward/backward → 메모리는 1/grad_accum, 결과는 같은 batch
        opt.zero_grad()
        micro = max(1, batch_size // args.grad_accum)
        total = 0.0
        for _ in range(args.grad_accum):
            x, y = load_batch(step, micro, device, true_w, corrupt_at=fail_at if args.fail == "crash" else -1,
                              shard=shard, bad_shards=bad_shards)
            part = nn.functional.mse_loss(model(x), y) / args.grad_accum
            part.backward()
            total += part.item()
        opt.step()
        loss = torch.tensor(total)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        time.sleep(max(0.0, args.step_time * (args.slow_factor if slow else 1) - (time.time() - t0)))

        if step % args.log_every == 0:
            mem = torch.cuda.memory_allocated(device) / 2 ** 20 if device.type == "cuda" else 0
            log.info(f"step {step}/{args.steps} loss {loss.item():.4g} lr {opt.param_groups[0]['lr']:.1e} "
                     f"gpu_mem {mem:.0f}MiB step_time {time.time() - t0:.2f}s")
        if step % args.ckpt_every == 0:
            # step별 파일로 남긴다 → loss가 망가지기 "전" 체크포인트로 돌아갈 수 있다
            ck = {"step": step, "model": model.state_dict()}
            torch.save(ck, ckpt_path)
            torch.save(ck, ckpt_file(job_dir, step))
            log.info(f"Saved checkpoint at step {step}")

    log.info("Training finished. Iteration limit reached")
    if args.fail == "hang_at_shutdown":
        # 데이터는 다 저장됐는데 종료(예: 환경/서버 연결 정리)에서 멈춘 상황. 네 Voyager run과 같은 증상.
        log.info("Shutting down environment...")
        hang_forever()
    log.info("Exit 0")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        # 예외로 죽을 때 traceback을 로그 파일에도 남긴다 (실제 학습 코드/런처도 보통 이렇게 한다)
        logging.getLogger("train").exception("Training crashed with an unhandled exception")
        sys.exit(1)
