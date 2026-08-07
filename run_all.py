import argparse
import subprocess
import sys
import time


EXPERIMENTS = (
    ("567.png (UNet)", "main.py"),
    ("123.png (UNet)", "main_123.py"),
    ("123.png (ProjectedUNet)", "main_projected_123.py"),
)


def run_one(label, script, iterations, seed):
    cmd = [sys.executable, script, "--iterations", str(iterations)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    print("=" * 72)
    print("START  {}  ->  {}".format(label, " ".join(cmd)))
    print("=" * 72)
    start = time.time()
    code = subprocess.call(cmd)
    print("\n>>> {} {} in {:.1f}s\n".format(
        label, "OK" if code == 0 else "FAILED (exit {})".format(code), time.time() - start))
    return code


def main():
    parser = argparse.ArgumentParser(description="Run all three phase-retrieval experiments in sequence")
    parser.add_argument("-n", "--iterations", type=int, default=3000,
                        help="shared iteration count for all three experiments (default 3000)")
    parser.add_argument("--seed", type=int, default=None, help="optional shared random seed")
    args = parser.parse_args()

    print("Running {} experiments, each for {} iterations\n".format(len(EXPERIMENTS), args.iterations))
    overall = time.time()
    failures = 0
    for label, script in EXPERIMENTS:
        if run_one(label, script, args.iterations, args.seed) != 0:
            failures += 1

    print("#" * 72)
    print("ALL DONE in {:.1f}s   ({}/{})".format(
        time.time() - overall, len(EXPERIMENTS) - failures, len(EXPERIMENTS)))
    print("#" * 72)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
