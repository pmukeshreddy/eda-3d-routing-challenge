"""Reproducible standard-library RL training and held-out routing experiments.

python -m m3d.rl train --out-dir experiments/rl_hard_pilot --budget-seconds 1800
python -m m3d.rl evaluate --policy experiments/rl_hard_pilot/training/policy.json \
    --out-dir experiments/rl_hard_pilot/evaluation
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

from .checker import check
from .generator import generate_feasible
from .model import Instance, Submission
from .negotiated import route_negotiated
from .rl_policy import LinearPolicy, PolicySelector, FEATURE_DIM, excess_occupancy
from .suite import suite_configs


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def append_json(path, data):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(data, allow_nan=False) + '\n')


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


def instance_fingerprint(inst):
    data = inst.to_dict()
    for field in ('name', 'seed', 'master_seed', 'params', 'format', 'version'):
        data.pop(field, None)
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def assert_disjoint(groups):
    seen = {}
    for split, fingerprints in groups.items():
        for fingerprint in fingerprints:
            if fingerprint in seen:
                raise ValueError(f'dataset overlap between {seen[fingerprint]} and {split}')
            seen[fingerprint] = split


def suite_cases(suite):
    suite = Path(suite)
    manifest = json.loads((suite / 'suite.json').read_text())
    cases = []
    for entry in manifest['cases']:
        inst = Instance.load(str(suite / entry['instance_file']))
        reference = Submission.load(str(suite / entry['reference_file']))
        checked = check(inst, reference)
        if not checked.legal or checked.total_delay != entry['baseline_total']:
            raise ValueError(f'invalid reference for {inst.name}')
        cases.append({'instance': inst, 'baseline_total': entry['baseline_total'],
                      'fingerprint': instance_fingerprint(inst)})
    return cases


def prepare_dataset(directory, heldout_suite, seed=42):
    directory = Path(directory)
    if (directory / 'manifest.json').exists():
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['seed'] != seed:
            raise ValueError('existing dataset uses a different seed; choose another output directory')
        load_dataset(directory, heldout_suite)
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    heldout = [c['fingerprint'] for c in suite_cases(heldout_suite)]
    manifest = {'format': 'm3d-rl-dataset', 'seed': seed, 'train': [], 'validation': [],
                'heldout_fingerprints': heldout}
    for split, repeats, salt in [('train', 2, 1000000), ('validation', 1, 2000000)]:
        for repetition in range(repeats):
            for index, cfg in enumerate(suite_configs('hard', master_seed=salt + seed + repetition * 10000)):
                cfg.name = f'{split}_{repetition * 9 + index + 1:02d}'
                case_started = time.perf_counter()
                result = generate_feasible(cfg)
                result.instance.save(str(directory / f'{cfg.name}.json'))
                result.reference.save(str(directory / f'{cfg.name}.sol.json'))
                entry = {'file': f'{cfg.name}.json', 'reference': f'{cfg.name}.sol.json',
                         'baseline_total': result.baseline_total,
                         'fingerprint': instance_fingerprint(result.instance),
                         'requested_config': cfg.to_params(), 'actual_width': result.instance.width,
                         'generation_attempts': result.attempts,
                         'generation_seconds': time.perf_counter() - case_started}
                manifest[split].append(entry)
                print(f'data {cfg.name}: {result.instance.width}x{result.instance.height}, '
                      f'baseline={result.baseline_total}, attempts={result.attempts}, '
                      f'{entry["generation_seconds"]:.1f}s', flush=True)
    assert_disjoint({'train': [e['fingerprint'] for e in manifest['train']],
                     'validation': [e['fingerprint'] for e in manifest['validation']],
                     'test': heldout})
    manifest['generation_seconds'] = time.perf_counter() - started
    write_json(directory / 'manifest.json', manifest)
    return directory


def load_dataset(directory, heldout_suite=None):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    result = {'manifest': manifest, 'train': [], 'validation': []}
    fingerprints = {}
    for split in ('train', 'validation'):
        if not manifest[split]:
            raise ValueError(f'{split} split must not be empty')
        fingerprints[split] = []
        for entry in manifest[split]:
            inst = Instance.load(str(directory / entry['file']))
            fingerprint = instance_fingerprint(inst)
            if fingerprint != entry['fingerprint']:
                raise ValueError(f'instance fingerprint mismatch: {entry["file"]}')
            reference = Submission.load(str(directory / entry['reference']))
            checked = check(inst, reference)
            if not checked.legal or checked.total_delay != entry['baseline_total']:
                raise ValueError(f'invalid dataset reference: {entry["file"]}')
            fingerprints[split].append(fingerprint)
            result[split].append({'instance': inst, 'baseline_total': entry['baseline_total'],
                                  'fingerprint': fingerprint})
    fingerprints['test'] = ([c['fingerprint'] for c in suite_cases(heldout_suite)]
                            if heldout_suite else manifest.get('heldout_fingerprints', []))
    assert_disjoint(fingerprints)
    return result


def terminal_reward(baseline_delay, routed_delay):
    if routed_delay is None:
        return -1.0
    return 1.0 + baseline_delay / (baseline_delay + routed_delay)


def summarize(rows):
    cases = []
    for original in rows:
        row = dict(original)
        legal = row['legal']
        row['ratio'] = row['baseline_delay'] / row['total_delay'] if legal else None
        row['delay_reduction_pct'] = (100.0 * (row['baseline_delay'] - row['total_delay']) /
                                      row['baseline_delay']) if legal else None
        cases.append(row)
    complete = bool(cases) and all(row['legal'] for row in cases)
    total = sum(row['total_delay'] for row in cases) if complete else None
    baseline = sum(row['baseline_delay'] for row in cases)
    return {'complete': complete, 'n_legal': sum(row['legal'] for row in cases),
            'n_cases': len(cases), 'baseline_total': baseline, 'total_delay': total,
            'total_delay_reduction_pct': 100.0 * (baseline - total) / baseline if complete else None,
            'aggregate_score': math.exp(sum(math.log(row['ratio']) for row in cases) / len(cases))
            if complete else 0.0,
            'total_runtime_s': sum(row['runtime_s'] for row in cases), 'cases': cases}


def run_method(case, method, policy=None, seed=42, training=False, rng=None):
    inst = case['instance']
    selector = None
    if method == 'negotiated_rl':
        if policy is None:
            raise ValueError('RL routing requires a policy')
        if isinstance(policy, LinearPolicy):
            selector = PolicySelector(policy, rng or random.Random(seed), training=training)
        else:
            from .ppo import NeuralSelector
            selector = NeuralSelector(policy, training=training, seed=seed)
    elif method == 'random':
        action_rng = random.Random(seed)
        selector = lambda router, eligible, pressure, iteration: action_rng.choice(eligible)
    elif method == 'most_conflicted':
        selector = lambda router, eligible, pressure, iteration: max(
            eligible, key=lambda nid: (excess_occupancy(router, nid), -nid))
    elif method not in ('negotiated', 'negotiated2'):
        raise ValueError(f'unknown method: {method}')
    started = time.perf_counter()
    if method == 'negotiated2':
        sub, stats, best_delay = None, None, None
        for order in ('bbox_desc', 'id'):
            candidate, candidate_stats = route_negotiated(inst, order=order)
            checked = check(inst, candidate) if candidate is not None else None
            if checked is not None and checked.legal and (best_delay is None or checked.total_delay < best_delay):
                sub, stats, best_delay = candidate, candidate_stats, checked.total_delay
    else:
        sub, stats = route_negotiated(inst, selector=selector)
    elapsed = time.perf_counter() - started
    checked = check(inst, sub) if sub is not None else None
    legal = checked is not None and checked.legal
    row = {'case': inst.name, 'legal': legal, 'baseline_delay': case['baseline_total'],
           'total_delay': checked.total_delay if legal else None,
           'runtime_s': elapsed, 'stats': asdict(stats) if stats is not None else None,
           'attempts': 2 if method == 'negotiated2' else 1}
    row['reward'] = terminal_reward(case['baseline_total'], row['total_delay'])
    return sub, row, selector


def validate(policy, cases):
    rows = [run_method(case, 'negotiated_rl', policy=policy)[1] for case in cases]
    result = summarize(rows)
    result['mean_reward'] = sum(row['reward'] for row in rows) / len(rows)
    return result


def train_dataset(dataset, out_dir, budget_seconds=1800, seed=42, max_episodes=None):
    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError('budget_seconds must be positive and finite')
    if max_episodes is not None and max_episodes < 1:
        raise ValueError('max_episodes must be positive')
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError('training output must be empty; choose another output directory')
    (out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    policy = LinearPolicy()
    action_rng, shuffle_rng = random.Random(seed), random.Random(seed + 1)
    configuration = {'algorithm': 'linear-softmax-REINFORCE', 'seed': seed,
                     'budget_seconds': budget_seconds, 'batch_size': 4, 'learning_rate': 0.05,
                     'gradient_norm_cap': 1.0, 'return_baseline': 1.5,
                     'max_iters': 50, 'validation_interval': 20,
                     'train_fingerprints': [c['fingerprint'] for c in dataset['train']],
                     'validation_fingerprints': [c['fingerprint'] for c in dataset['validation']],
                     'source_hashes': source_hashes(), 'python': sys.version,
                     'platform': platform.platform()}
    write_json(out_dir / 'config.json', configuration)
    pending_gradients, queue = [], []
    episodes = updates = 0
    best_rank = None
    best_validation = None
    selected_episode = None
    started = time.perf_counter()

    def checkpoint():
        nonlocal best_rank, best_validation, selected_episode
        metadata = dict(configuration, episodes=episodes, updates=updates)
        policy.save(out_dir / 'checkpoints' / f'episode_{episodes:05d}.json', metadata)
        validation = validate(policy, dataset['validation'])
        rank = (validation['n_legal'], validation['mean_reward'])
        append_json(out_dir / 'validation.jsonl', {'episodes': episodes, 'updates': updates,
                    'elapsed_s': time.perf_counter() - started, 'result': validation})
        if best_rank is None or rank > best_rank:
            best_rank, best_validation, selected_episode = rank, validation, episodes
            policy.save(out_dir / 'policy.json', dict(metadata, validation=validation))
        print(f'validation episode={episodes}: legal={validation["n_legal"]}/{len(dataset["validation"])}, '
              f'score={validation["aggregate_score"]:.6f}, selected_episode={selected_episode}', flush=True)

    while (episodes == 0 or time.perf_counter() - started < budget_seconds):
        if max_episodes is not None and episodes >= max_episodes:
            break
        if not queue:
            queue = list(range(len(dataset['train'])))
            shuffle_rng.shuffle(queue)
        case = dataset['train'][queue.pop()]
        _, row, selector = run_method(case, 'negotiated_rl', policy, training=True, rng=action_rng)
        advantage = row['reward'] - 1.5
        pending_gradients.append([advantage * g for g in selector.gradient])
        episodes += 1
        row.update(episode=episodes, decisions=selector.decisions, advantage=advantage,
                   elapsed_s=time.perf_counter() - started)
        append_json(out_dir / 'episodes.jsonl', row)
        if len(pending_gradients) == 4:
            policy.update(pending_gradients)
            pending_gradients.clear()
            updates += 1
        print(f'train episode={episodes}: case={case["instance"].name}, legal={row["legal"]}, '
              f'delay={row["total_delay"]}, reward={row["reward"]:.5f}, '
              f'elapsed={row["elapsed_s"]:.1f}s', flush=True)
        if episodes % 20 == 0:
            checkpoint()
    if pending_gradients:
        policy.update(pending_gradients)
        updates += 1
    training_elapsed = time.perf_counter() - started
    if episodes % 20 != 0:
        checkpoint()
    result = {'episodes': episodes, 'updates': updates, 'selected_episode': selected_episode,
              'training_and_intermediate_validation_s': training_elapsed,
              'including_final_validation_s': time.perf_counter() - started,
              'validation': best_validation, 'configuration': configuration}
    write_json(out_dir / 'training.json', result)
    return result


def evaluate_suite(suite, policy_path, out_dir, seed=42):
    cases = suite_cases(suite)
    if Path(policy_path).suffix == '.pt':
        import torch
        from .ppo import load_checkpoint
        torch.set_num_threads(1)
        policy, metadata = load_checkpoint(policy_path)
    else:
        policy = LinearPolicy.load(policy_path)
        metadata = json.loads(Path(policy_path).read_text()).get('metadata', {})
    assert_disjoint({'train': metadata.get('train_fingerprints', []),
                     'validation': metadata.get('validation_fingerprints', []),
                     'test': [c['fingerprint'] for c in cases]})
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError('evaluation output must be empty; choose another output directory')
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = ['negotiated', 'negotiated_rl', 'random', 'most_conflicted', 'negotiated2']
    report = {'suite': str(suite), 'seed': seed, 'python': sys.version,
              'platform': platform.platform(), 'source_hashes': source_hashes(),
              'policy_sha256': hashlib.sha256(Path(policy_path).read_bytes()).hexdigest(),
              'policy_metadata': metadata, 'methods': {}}
    for method in methods:
        directory = out_dir / method
        directory.mkdir()
        rows, runtimes = [], {}
        for index, case in enumerate(cases):
            sub, row, _ = run_method(case, method, policy=policy, seed=seed + index)
            rows.append(row)
            runtimes[row['case']] = row['runtime_s']
            if sub is not None:
                sub.save(str(directory / f'{row["case"]}.sol.json'))
            append_json(directory / 'cases.jsonl', row)
            write_json(directory / 'runtime.json', runtimes)
            print(f'eval {method} {row["case"]}: legal={row["legal"]}, '
                  f'delay={row["total_delay"]}, runtime={row["runtime_s"]:.2f}s', flush=True)
        report['methods'][method] = summarize(rows)
        write_json(out_dir / 'results.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train', help='generate independent data and train a rerouting policy')
    train.add_argument('--suite', default='benchmarks_hard', help='held-out suite, never used to train')
    train.add_argument('--out-dir', required=True)
    train.add_argument('--budget-seconds', type=float, default=1800)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--max-episodes', type=int, default=None)
    evaluate = commands.add_parser('evaluate', help='evaluate frozen policy and four controls')
    evaluate.add_argument('--suite', default='benchmarks_hard')
    evaluate.add_argument('--policy', required=True)
    evaluate.add_argument('--out-dir', required=True)
    evaluate.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == 'train':
        root = Path(args.out_dir)
        data = prepare_dataset(root / 'data', args.suite, args.seed)
        result = train_dataset(load_dataset(data, args.suite), root / 'training',
                               args.budget_seconds, args.seed, args.max_episodes)
        print(json.dumps({key: result[key] for key in ('episodes', 'updates', 'selected_episode')}, indent=2))
    else:
        report = evaluate_suite(args.suite, args.policy, args.out_dir, args.seed)
        for name, result in report['methods'].items():
            print(f'{name}: {result["n_legal"]}/{result["n_cases"]} legal, '
                  f'score={result["aggregate_score"]:.6f}, delay={result["total_delay"]}, '
                  f'reduction={result["total_delay_reduction_pct"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
