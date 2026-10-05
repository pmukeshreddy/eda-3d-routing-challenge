"""Load one training input and take one environment action; not an RL rollout."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m3d.routing_data import TrainingSampler
from m3d.routing_env import RoutingBudget, RoutingEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="build/data/routing_v1")
    args = parser.parse_args()
    instance, reward_metadata = TrainingSampler(args.data, seed=0).sample()
    with RoutingEnv() as env:
        initial = env.reset(instance, seed=0, budget=RoutingBudget(max_seconds=10, max_calls=1))
        assert initial["routes"] == {} and initial["vertex_owners"] == {}
        result = env.step(env.eligible_net_ids()[0])
        report = {"case": instance.name, "empty_reset": True,
                  "selected_net": result.diagnostics["net_id"],
                  "engine_status": result.diagnostics["engine_status"],
                  "route_installed": result.diagnostics["route_installed"],
                  "engine_calls": result.observation["budget"]["engine_calls"],
                  "reward_metadata_separate_from_instance": asdict(reward_metadata),
                  "scope": "one-action data integration check; not Block 2 hard-tier acceptance"}
        print(json.dumps(report, indent=2))
        return 0 if report["route_installed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
