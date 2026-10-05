"""Run the fixed Block 4 GRPO experiment (or its tiny mechanics check)."""
import argparse
import json
from pathlib import Path
from time import perf_counter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="build/data/routing_v1")
    parser.add_argument("--out", default="build/block4")
    parser.add_argument("--config", default="configs/scheduler_grpo_v1.json")
    parser.add_argument("--mechanics", action="store_true")
    args = parser.parse_args(argv)
    # Lazy import keeps spawned native routing workers free of PyTorch/model
    # allocations on machines with shared GPU memory.
    from .scheduler_training import mechanics, train, write_json
    configuration = json.loads(Path(args.config).read_text())
    started = perf_counter()
    try:
        result = (mechanics if args.mechanics else train)(args.data, args.out, configuration)
    except TimeoutError as error:
        # Previously saved groups, metrics and checkpoints stay intact. A
        # deadline cannot turn a partial experiment into an acceptance claim.
        result = {"status": "partial", "stop_reason": str(error), "runtime_s": perf_counter() - started,
                  "wall_cap_s": configuration["mechanics"]["wall_seconds"] if args.mechanics else configuration["wall_seconds"],
                  "artifacts_preserved": True, "test_instances_opened": 0,
                  "block_2_gap": "Unchanged: case_05 had 3 vertex conflicts after 50 passes."}
        write_json(Path(args.out) / "acceptance.json", result)
        print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
