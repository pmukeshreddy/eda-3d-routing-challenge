"""Block 4 validation-only evaluation. Test routing is reserved for Block 5."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="build/data/routing_v1")
    parser.add_argument("--split", required=True, choices=("val",))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", help="optional new validation JSONL; existing files are not overwritten")
    args = parser.parse_args(argv)
    import tempfile
    import torch
    from .routing_data import load_split
    from .scheduler_grpo import load_checkpoint
    from .scheduler_training import choose_device, evaluate_policy
    _, checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    torch.set_num_threads(config["cpu_threads"])
    device = choose_device(config["device"])
    policy, _ = load_checkpoint(args.checkpoint, device=device)
    validation = load_split(args.data, split=args.split)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(args.out) if args.out else Path(directory) / "validation.jsonl"
        if output.exists():
            raise ValueError("evaluation output already exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        result = evaluate_policy(policy, validation, config, checkpoint["failure_floor"], device,
                                 label="checkpoint", update=checkpoint["update"], output=output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
