import argparse
import glob
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--nproc", type=int, default=2)
    args, extra = parser.parse_known_args()

    if args.ckpt is None:
        candidates = sorted(glob.glob(os.path.join(args.ckpt_dir, "ckpt_step*.pt")))
        if not candidates:
            print(f"no_checkpoint_found dir={args.ckpt_dir}")
            sys.exit(1)
        args.ckpt = candidates[-1]

    print(f"resuming from {args.ckpt}")
    train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    cmd = [
        "torchrun", f"--nproc_per_node={args.nproc}",
        train_script, "--resume", args.ckpt, "--ckpt-dir", args.ckpt_dir,
    ] + extra
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
