"""Freeze, evaluate once, or verify the Block 5 artifacts without training."""
import argparse
from time import perf_counter


def main(argv=None):
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("freeze", "evaluate", "verify"))
    parser.add_argument("--out", default="build/block5")
    args = parser.parse_args(argv)
    # Spawned routing workers import this module without importing PyTorch.
    from ._final_evaluation import freeze, evaluate, verify
    return {"freeze": freeze, "evaluate": evaluate, "verify": verify}[args.phase](args.out, started=started)


if __name__ == "__main__":
    raise SystemExit(main())
