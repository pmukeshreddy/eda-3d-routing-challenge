"""Focused GRPO/data/checkpoint contracts; no upstream or held-out routing."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from m3d.scheduler_grpo import (group_advantages, policy_loss, failure_floor,
                                TrainingCases, save_checkpoint, load_checkpoint)


class SchedulerGRPOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_group_advantages_and_collapse(self):
        advantages = group_advantages([1., 2., 3.])
        torch.testing.assert_close(advantages, torch.tensor([-math.sqrt(1.5), 0., math.sqrt(1.5)]))
        self.assertEqual(group_advantages([-5., -5., -5., -5.]).tolist(), [0.] * 4)
        with self.assertRaises(ValueError):
            group_advantages([1., float("nan")])

    def test_clipping_entropy_kl_and_finite_masked_gradients(self):
        # Selected action's ratio is 1.8; positive advantage clips to 1.2.
        logits = torch.tensor([[math.log(.9), math.log(.1), 50.]], requires_grad=True)
        old = torch.tensor([[math.log(.5), math.log(.5), 0.]])
        eligible = torch.tensor([[True, True, False]])
        terms = policy_loss(logits, eligible, torch.tensor([0]), torch.tensor([math.log(.5)]),
                            old, torch.tensor([1.]), clip_ratio=.2, entropy_coefficient=0., kl_coefficient=0.)
        self.assertAlmostEqual(terms["loss"].item(), -1.2, places=5)
        self.assertAlmostEqual(terms["kl"].item(), .5 * math.log(.5/.9) + .5 * math.log(.5/.1), places=5)
        terms["loss"].sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 2].item(), 0.)
        negative = policy_loss(logits.detach(), eligible, torch.tensor([0]), torch.tensor([math.log(.5)]),
                              old, torch.tensor([-1.]), clip_ratio=.2, entropy_coefficient=0., kl_coefficient=0.)
        self.assertAlmostEqual(negative["loss"].item(), 1.8, places=5)

    def test_failure_floor_uses_metadata_and_training_access_is_explicit(self):
        root = Path(__file__).resolve().parents[1]
        manifest = {"configuration": json.loads((root / "configs/routing_data_v1.json").read_text()),
                    "cases": [{"split": "test", "reference_total_delay": 123}]}
        first = failure_floor(manifest)
        changed = copy.deepcopy(manifest)
        # No per-case (including held-out) metadata is needed to define floor.
        changed["cases"] = []
        self.assertEqual(first, failure_floor(changed))
        self.assertLess(first["reward"], -math.log(first["legal_delay_upper_bound"]))
        accessed = []
        class FakeSplit:
            entries = ({"case_id": "training-only"},)
            def __len__(self): return 1
            def load(self, i): accessed.append(i); return "instance", "metadata"
        def split_loader(path, *, split):
            self.assertEqual(split, "train")
            return FakeSplit()
        with patch("m3d.scheduler_grpo.load_split", split_loader):
            data = TrainingCases("unused", seed=5)
            self.assertEqual(data.next(), ("instance", "metadata"))
        self.assertEqual(accessed, [0])

    def test_checkpoint_round_trip_preserves_model_optimizer_and_actions(self):
        from m3d.scheduler_policy import SchedulerPolicy
        from m3d.scheduler_features import EncodedState, NET_FEATURES, GLOBAL_FEATURES
        torch.manual_seed(42)
        policy = SchedulerPolicy()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=.0003)
        state = EncodedState((8, 42, 90), torch.randn(3, len(NET_FEATURES)),
                             torch.randn(len(GLOBAL_FEATURES)), torch.eye(3),
                             torch.tensor([False, True, True]))
        action = policy.act(state, deterministic=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            config = {"model": {}, "seeds": {"model": 42}}
            save_checkpoint(path, policy, optimizer, 2, config, {"reward": -20.}, extra={"test": True})
            loaded, checkpoint = load_checkpoint(path, device="cpu")
            self.assertEqual(loaded.act(state, deterministic=True), action)
            self.assertEqual(checkpoint["update"], 2)
            self.assertEqual(checkpoint["optimizer"]["param_groups"], optimizer.state_dict()["param_groups"])
            for name, parameter in policy.state_dict().items():
                torch.testing.assert_close(parameter, loaded.state_dict()[name], rtol=0, atol=0)

    def test_wall_deadline_refuses_to_shrink_episode_budget(self):
        from m3d.scheduler_rollout import collect_rollouts
        from tests.test_scheduler_policy import observation
        job = {"instance": observation()["instance"], "action_seed": 5}
        config = {"concurrency": 1, "routing_budget": {"max_seconds": 120}}
        with patch("m3d.scheduler_rollout.RoutingEnv") as env:
            with self.assertRaises(TimeoutError):
                collect_rollouts([job], selector="bbox", policy=None, config=config,
                                 floor={"reward": -20.}, device="cpu", deadline=0.)
            env.return_value.reset.assert_not_called()
            env.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
