import importlib.util
import tempfile
import unittest
from pathlib import Path

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH_AVAILABLE, 'install requirements-rl.txt for neural RL tests')
class TestPPO(unittest.TestCase):
    def setUp(self):
        import torch
        from m3d.ppo import GraphActorCritic
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.torch = torch
        self.model = GraphActorCritic()

    def test_action_mask_and_permutation_equivariance(self):
        t = self.torch
        x = t.rand(1, 4, 30)
        graph = t.eye(4, dtype=t.bool).unsqueeze(0)
        graph[0, 0, 1] = graph[0, 1, 0] = True
        mask = t.tensor([[True, False, True, True]])
        logits, value = self.model(x, graph, mask)
        self.assertTrue(t.isneginf(logits[0, 1]))
        permutation = t.tensor([2, 0, 3, 1])
        reordered, other_value = self.model(x[:, permutation], graph[:, permutation][:, :, permutation], mask[:, permutation])
        t.testing.assert_close(reordered, logits[:, permutation])
        t.testing.assert_close(other_value, value)

    def test_padding_does_not_change_real_nodes(self):
        from m3d.ppo import pack_observations
        import numpy as np
        short = {'features': np.ones((2, 30), dtype=np.float32),
                 'graph': np.eye(2, dtype=bool), 'eligible': np.array([True, False])}
        long = {'features': np.zeros((5, 30), dtype=np.float32),
                'graph': np.eye(5, dtype=bool), 'eligible': np.ones(5, dtype=bool)}
        a, av = self.model(*pack_observations([short]))
        b, bv = self.model(*pack_observations([short, long]))
        self.torch.testing.assert_close(a[0], b[0, :2])
        self.torch.testing.assert_close(av[0], bv[0])
        self.assertFalse(self.torch.isnan(b).any())

    def test_complete_episode_returns_keep_terminal_credit(self):
        from m3d.ppo import episode_advantages
        advantage, returns = episode_advantages([0.2, 0.5, 0.8], 1.5)
        self.assertEqual(returns, [1.5, 1.5, 1.5])
        for actual, expected in zip(advantage, [1.3, 1.0, 0.7]):
            self.assertAlmostEqual(actual, expected)

    def test_clipped_policy_objective_and_gradients(self):
        from m3d.ppo import clipped_policy_loss
        t = self.torch
        logp = t.tensor([1.5, 0.5]).log().requires_grad_()
        loss = clipped_policy_loss(logp, t.zeros(2), t.tensor([1., -1.]), clip=0.2)
        self.assertAlmostEqual(loss.item(), -0.2, places=6)
        loss.backward()
        t.testing.assert_close(logp.grad, t.zeros(2))

    def test_checkpoint_preserves_predictions_and_feature_contract(self):
        from m3d.ppo import save_checkpoint, load_checkpoint
        t = self.torch
        observation = (t.rand(1, 3, 30), t.eye(3, dtype=t.bool)[None], t.ones(1, 3, dtype=t.bool))
        expected = self.model(*observation)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.pt'
            save_checkpoint(path, self.model, {'seed': 42})
            restored, metadata = load_checkpoint(path)
            self.assertEqual(metadata['seed'], 42)
            actual = restored(*observation)
            for a, b in zip(expected, actual):
                t.testing.assert_close(a, b)

    def test_real_episode_collection_and_ppo_update(self):
        from m3d.generator import GenConfig, generate_feasible
        from m3d.ppo import ExpertSelector, NeuralSelector, ppo_update
        from m3d.ppo_train import collect_episode, save_expert, imitate
        from m3d.negotiated import route_negotiated
        from m3d.checker import check
        cfg = GenConfig(name='ppo_smoke', width=12, height=12, layers=6, n_nets=14,
                        frac_cross=.45, frac_local=.12, cell_min=2, cell_max=2,
                        pins_per_cell=2, cell_gap=0, seed=17, router='negotiated')
        generated = generate_feasible(cfg)
        case = {'instance': generated.instance, 'baseline_total': generated.baseline_total}
        teacher = ExpertSelector()
        sub, _ = route_negotiated(generated.instance, selector=teacher)
        self.assertEqual(check(generated.instance, sub).total_delay, generated.baseline_total)
        self.assertTrue(teacher.steps)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'expert.npz'
            save_expert(path, teacher.steps, len(generated.instance.nets))
            history = imitate(self.model, [path], epochs=1, seed=42)
            self.assertGreater(history[0]['decisions'], 0)
        episode = collect_episode(self.model, case, seed=42, training=True)
        self.assertTrue(episode['steps'])
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=3e-4)
        result = ppo_update(self.model, optimizer, [episode], epochs=1, minibatch=16)
        self.assertGreater(result['minibatches'], 0)
        self.assertTrue(any(not self.torch.equal(before[name], p)
                            for name, p in self.model.named_parameters()))

    def test_training_driver_parallel_rollout_selection_and_completed_resume(self):
        import json
        from m3d.generator import GenConfig, generate_feasible
        from m3d.ppo import ExpertSelector, load_checkpoint
        from m3d.ppo_train import train_seed, save_expert
        from m3d.negotiated import route_negotiated
        from m3d.rl import instance_fingerprint
        dataset = {'train': [], 'validation': [], 'manifest': {'train': []}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, seed in [('train', 17), ('validation', 18)]:
                cfg = GenConfig(name=split, width=12, height=12, layers=6, n_nets=14,
                                frac_cross=.45, frac_local=.12, cell_min=2, cell_max=2,
                                pins_per_cell=2, cell_gap=0, seed=seed, router='negotiated')
                generated = generate_feasible(cfg)
                dataset[split].append({'instance': generated.instance,
                    'baseline_total': generated.baseline_total,
                    'fingerprint': instance_fingerprint(generated.instance)})
                if split == 'train':
                    expert = ExpertSelector()
                    route_negotiated(generated.instance, selector=expert)
                    save_expert(root / 'expert.npz', expert.steps, len(generated.instance.nets))
                    dataset['manifest']['train'].append({'expert_file': 'expert.npz'})
            kwargs = dict(workers=1, max_updates=1, min_updates=0, patience=1,
                          validation_interval=1, batch_episodes=2, imitation_epochs=1)
            result = train_seed(root, dataset, root / 'run', 42, **kwargs)
            self.assertEqual(result['updates'], 1)
            self.assertEqual(result['episodes'], 2)
            _, metadata = load_checkpoint(root / 'run/policy.pt')
            self.assertEqual(metadata['seed'], 42)
            self.assertIn(metadata['stage'], ('imitation', 'ppo'))
            validation = [json.loads(line) for line in (root / 'run/validation.jsonl').read_text().splitlines()]
            self.assertEqual([v['update'] for v in validation], [0, 1])
            again = train_seed(root, dataset, root / 'run', 42, **kwargs)
            self.assertEqual(again, result)
            configuration = json.loads((root / 'run/config.json').read_text())
            configuration['source_hashes']['ppo.py'] = 'changed-algorithm'
            (root / 'run/config.json').write_text(json.dumps(configuration))
            with self.assertRaisesRegex(ValueError, 'configuration'):
                train_seed(root, dataset, root / 'run', 42, **kwargs)


if __name__ == '__main__':
    unittest.main()
