import copy
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from m3d.generator import GenConfig, generate_feasible
from m3d.model import Instance
from m3d.rl import (terminal_reward, summarize, instance_fingerprint,
                    assert_disjoint, train_dataset, load_dataset)
from m3d.rl_policy import LinearPolicy


class TestExperiment(unittest.TestCase):
    def test_run_suite_rejects_stale_solution_artifacts(self):
        from m3d.cli import cmd_run_suite
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'case_01.sol.json').write_text('{}')
            # This must be rejected before any suite file is opened or routing starts.
            args = Namespace(router='negotiated', out_dir=directory,
                             suite='does-not-exist', policy=None)
            with self.assertRaisesRegex(ValueError, 'empty'):
                cmd_run_suite(args)

    def test_reward_prioritizes_legality_and_lower_real_delay(self):
        self.assertEqual(terminal_reward(100, None), -1.0)
        self.assertEqual(terminal_reward(100, 100), 1.5)
        self.assertGreater(terminal_reward(100, 80), terminal_reward(100, 100))
        self.assertGreater(terminal_reward(100, 1000000), terminal_reward(100, None))

    def test_report_percentages_and_incomplete_score(self):
        rows = [dict(case='a', legal=True, baseline_delay=100, total_delay=80, runtime_s=2),
                dict(case='b', legal=True, baseline_delay=200, total_delay=220, runtime_s=3)]
        result = summarize(rows)
        self.assertEqual(result['total_delay'], 300)
        self.assertEqual(result['total_delay_reduction_pct'], 0)
        self.assertAlmostEqual(result['aggregate_score'], (1.25 * 200 / 220) ** 0.5)
        self.assertEqual(result['cases'][0]['delay_reduction_pct'], 20)
        self.assertEqual(result['cases'][1]['delay_reduction_pct'], -10)
        rows[1].update(legal=False, total_delay=None)
        result = summarize(rows)
        self.assertEqual(result['aggregate_score'], 0)
        self.assertIsNone(result['total_delay_reduction_pct'])
        self.assertIsNone(result['total_delay'])

    def test_fingerprint_ignores_labels_and_detects_holdout_leakage(self):
        root = Path(__file__).resolve().parents[1]
        inst = Instance.load(str(root / 'benchmarks_hard/case_01.json'))
        altered = copy.deepcopy(inst)
        altered.name = 'training_name'
        altered.seed = 123
        altered.params = {}
        fingerprint = instance_fingerprint(inst)
        self.assertEqual(fingerprint, instance_fingerprint(altered))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            assert_disjoint({'train': [fingerprint], 'validation': [], 'test': [fingerprint]})

    def test_smoke_training_saves_trained_selected_checkpoint(self):
        # Tiny real environments exercise collection, update, validation and reload.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            manifest = {'format': 'm3d-rl-dataset', 'seed': 42, 'train': [], 'validation': []}
            for split, seed in [('train', 17), ('validation', 18)]:
                cfg = GenConfig(name=split, width=12, height=12, layers=6, n_nets=14,
                                frac_cross=0.45, frac_local=0.12, cell_min=2, cell_max=2,
                                pins_per_cell=2, cell_gap=0, seed=seed, router='negotiated')
                generated = generate_feasible(cfg)
                generated.instance.save(str(data / f'{split}.json'))
                generated.reference.save(str(data / f'{split}.sol.json'))
                manifest[split].append({'file': f'{split}.json', 'reference': f'{split}.sol.json',
                                        'baseline_total': generated.baseline_total,
                                        'fingerprint': instance_fingerprint(generated.instance)})
            (data / 'manifest.json').write_text(json.dumps(manifest))
            dataset = load_dataset(data)
            result = train_dataset(dataset, root / 'run', budget_seconds=10,
                                   seed=42, max_episodes=4)
            self.assertEqual(result['episodes'], 4)
            self.assertEqual(result['updates'], 1)
            policy = LinearPolicy.load(root / 'run/policy.json')
            self.assertTrue(policy.weights)
            self.assertTrue((root / 'run/episodes.jsonl').exists())
            metadata = json.loads((root / 'run/policy.json').read_text())['metadata']
            self.assertGreater(metadata['episodes'], 0)
            self.assertEqual(len(result['validation']['cases']), 1)
            # Changes to generated instances cannot silently bypass split checks.
            manifest['validation'][0]['fingerprint'] = 'incorrect'
            (data / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                load_dataset(data)


if __name__ == '__main__':
    unittest.main()
