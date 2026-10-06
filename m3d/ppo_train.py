"""Dataset creation, baseline imitation, complete-episode PPO, and held-out evaluation.

Run: .venv/bin/python -m m3d.ppo_train run --out-dir experiments/ppo_hard
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import io
import json
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .checker import check
from .generator import generate_feasible
from .negotiated import route_negotiated
from .ppo import (GraphActorCritic, NeuralSelector, ExpertSelector, pack_observations,
                  ppo_update, save_checkpoint, load_checkpoint, state_bytes)
from .rl import (write_json, append_json, suite_cases, assert_disjoint, instance_fingerprint,
                 load_dataset, terminal_reward, summarize, source_hashes, evaluate_suite)
from .suite import suite_configs


def save_expert(path, steps, n_nets):
    np.savez_compressed(path,
        features=np.asarray([s['features'] for s in steps], dtype=np.float32).reshape(-1, n_nets, 30),
        graph=np.asarray([s['graph'] for s in steps], dtype=bool).reshape(-1, n_nets, n_nets),
        eligible=np.asarray([s['eligible'] for s in steps], dtype=bool).reshape(-1, n_nets),
        action=np.asarray([s['action'] for s in steps], dtype=np.int64))


def _generate_case(job):
    torch.set_num_threads(1)
    cfg, directory, split = job
    directory = Path(directory)
    metadata_path = directory / f'{cfg.name}.meta.json'
    if metadata_path.exists():
        entry = json.loads(metadata_path.read_text())
        if entry['requested_config'] != cfg.to_params():
            raise ValueError('existing dataset case has a different configuration')
        return split, entry
    started = time.perf_counter()
    generated = generate_feasible(cfg)
    generated.instance.save(str(directory / f'{cfg.name}.json'))
    generated.reference.save(str(directory / f'{cfg.name}.sol.json'))
    entry = {'file': f'{cfg.name}.json', 'reference': f'{cfg.name}.sol.json',
             'baseline_total': generated.baseline_total,
             'fingerprint': instance_fingerprint(generated.instance),
             'requested_config': cfg.to_params(), 'actual_width': generated.instance.width,
             'generation_attempts': generated.attempts}
    if split == 'train':
        expert = ExpertSelector()
        sub, _ = route_negotiated(generated.instance, selector=expert)
        verified = check(generated.instance, sub) if sub else None
        if verified is None or not verified.legal or verified.total_delay != generated.baseline_total:
            raise RuntimeError('baseline imitation did not reproduce the reference')
        entry['expert_file'] = f'{cfg.name}.expert.npz'
        entry['expert_decisions'] = len(expert.steps)
        save_expert(directory / entry['expert_file'], expert.steps, len(generated.instance.nets))
    entry['generation_seconds'] = time.perf_counter() - started
    write_json(metadata_path, entry)
    return split, entry


def prepare_data(directory, suite='benchmarks_hard', seed=42, workers=4,
                 train_repeats=16, validation_repeats=4):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    configuration = {'seed': seed, 'train_repeats': train_repeats,
                     'validation_repeats': validation_repeats}
    config_path = directory / 'data-config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError('existing dataset configuration differs; use another directory')
    write_json(config_path, configuration)
    if (directory / 'manifest.json').exists():
        load_dataset(directory, suite)
        return directory
    started = time.perf_counter()
    test_fingerprints = [c['fingerprint'] for c in suite_cases(suite)]
    manifest = {'format': 'm3d-rl-dataset', 'seed': seed, 'train': [], 'validation': [],
                'heldout_fingerprints': test_fingerprints, 'configuration': configuration}
    jobs = []
    for split, repeats, salt in [('train', train_repeats, 3000000),
                                 ('validation', validation_repeats, 4000000)]:
        for repetition in range(repeats):
            for index, cfg in enumerate(suite_configs('hard', master_seed=salt + seed + repetition * 10000)):
                cfg.name = f'{split}_{repetition * 9 + index + 1:03d}'
                jobs.append((cfg, str(directory), split))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(_generate_case, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            split, entry = future.result()
            manifest[split].append(entry)
            print(f'data {completed}/{len(jobs)} {entry["file"]}: '
                  f'baseline={entry["baseline_total"]}, width={entry["actual_width"]}, '
                  f'{entry["generation_seconds"]:.1f}s', flush=True)
    for split in ('train', 'validation'):
        manifest[split].sort(key=lambda entry: entry['file'])
    assert_disjoint({'train': [e['fingerprint'] for e in manifest['train']],
                     'validation': [e['fingerprint'] for e in manifest['validation']],
                     'test': test_fingerprints})
    manifest['generation_seconds'] = time.perf_counter() - started
    write_json(directory / 'manifest.json', manifest)
    return directory


def imitate(model, expert_paths, epochs=5, seed=42):
    rng = random.Random(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    history = []
    model.train()
    for epoch in range(epochs):
        paths = list(expert_paths)
        rng.shuffle(paths)
        count, correct, loss_sum = 0, 0, 0.
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in ('features', 'graph', 'eligible', 'action')}
            indices = list(range(len(arrays['action'])))
            rng.shuffle(indices)
            for start in range(0, len(indices), 128):
                batch = indices[start:start + 128]
                observations = [{key: arrays[key][i] for key in ('features', 'graph', 'eligible')}
                                for i in batch]
                target = torch.from_numpy(arrays['action'][batch])
                logits, values = model(*pack_observations(observations))
                loss = F.cross_entropy(logits, target) + .25 * F.mse_loss(values, torch.full_like(values, 1.5))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                optimizer.step()
                count += len(batch)
                correct += (logits.argmax(dim=1) == target).sum().item()
                loss_sum += loss.item() * len(batch)
        row = {'epoch': epoch + 1, 'decisions': count, 'accuracy': correct / max(1, count),
               'mean_loss': loss_sum / max(1, count)}
        history.append(row)
        print(f'imitation epoch={epoch + 1}: decisions={count}, accuracy={row["accuracy"]:.4f}', flush=True)
    return history


def collect_episode(model, case, seed=42, training=True):
    selector = NeuralSelector(model, training=training, seed=seed, record=training)
    started = time.perf_counter()
    sub, stats = route_negotiated(case['instance'], selector=selector)
    runtime = time.perf_counter() - started
    result = check(case['instance'], sub) if sub is not None else None
    legal = result is not None and result.legal
    row = {'case': case['instance'].name, 'legal': legal,
           'baseline_delay': case['baseline_total'],
           'total_delay': result.total_delay if legal else None,
           'runtime_s': runtime, 'stats': asdict(stats)}
    row['reward'] = terminal_reward(row['baseline_delay'], row['total_delay'])
    return {'row': row, 'steps': selector.steps}


_WORKER_DATA = None
_WORKER_MODEL = None
_WORKER_STATE = None


def _init_worker(dataset):
    global _WORKER_DATA, _WORKER_MODEL, _WORKER_STATE
    torch.set_num_threads(1)
    _WORKER_DATA = dataset
    _WORKER_MODEL = GraphActorCritic()
    _WORKER_STATE = None


def _worker_episode(job):
    global _WORKER_STATE
    weights, split, index, seed, training = job
    if _WORKER_STATE != weights:
        _WORKER_MODEL.load_state_dict(torch.load(io.BytesIO(weights), map_location='cpu', weights_only=True))
        _WORKER_STATE = weights
    return collect_episode(_WORKER_MODEL, _WORKER_DATA[split][index], seed, training)


def _validation(model, dataset, pool, seed):
    weights = state_bytes(model)
    jobs = [(weights, 'validation', i, seed + i, False) for i in range(len(dataset['validation']))]
    rows = [episode['row'] for episode in pool.map(_worker_episode, jobs)]
    result = summarize(rows)
    result['mean_reward'] = sum(row['reward'] for row in rows) / len(rows)
    return result


def train_seed(data_dir, dataset, out_dir, seed, workers=4, max_updates=200,
               min_updates=50, patience=8, validation_interval=10, batch_episodes=16,
               imitation_epochs=5):
    out_dir = Path(out_dir)
    (out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, workers))
    torch.manual_seed(seed)
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    model = GraphActorCritic()
    configuration = {'algorithm': 'PPO', 'seed': seed, 'model': 'conflict-graph-attention-64x2-4heads',
        'parameters': sum(p.numel() for p in model.parameters()), 'max_updates': max_updates,
        'min_updates': min_updates, 'patience': patience, 'validation_interval': validation_interval,
        'batch_episodes': batch_episodes, 'imitation_epochs': imitation_epochs, 'learning_rate': 3e-4,
        'gamma': 1., 'gae_lambda': 1., 'clip': .2, 'ppo_epochs': 4, 'minibatch': 128,
        'entropy_start': .01, 'entropy_end': .001, 'gradient_clip': .5, 'target_kl': .02,
        'workers': workers, 'torch_version': str(torch.__version__), 'python': sys.version,
        'train_fingerprints': [c['fingerprint'] for c in dataset['train']],
        'validation_fingerprints': [c['fingerprint'] for c in dataset['validation']],
        'source_hashes': source_hashes()}
    config_path = out_dir / 'config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != configuration:
            raise ValueError('training configuration differs from existing run')
    else:
        write_json(config_path, configuration)
    if (out_dir / 'training.json').exists():
        return json.loads((out_dir / 'training.json').read_text())
    warm_path = out_dir / 'imitation.pt'
    if warm_path.exists():
        model, _ = load_checkpoint(warm_path)
    else:
        paths = [Path(data_dir) / e['expert_file'] for e in dataset['manifest']['train']]
        history = imitate(model, paths, epochs=imitation_epochs, seed=seed)
        write_json(out_dir / 'imitation.json', history)
        save_checkpoint(warm_path, model, dict(configuration, stage='imitation', update=0))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    update, stale_checks, best_update = 0, 0, 0
    best_result, best_rank = None, None
    queue = []
    latest_path = out_dir / 'latest.pt'
    if latest_path.exists():
        saved = torch.load(latest_path, map_location='cpu', weights_only=True)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        update, stale_checks, best_update = saved['update'], saved['stale_checks'], saved['best_update']
        best_result, best_rank = saved['best_result'], tuple(saved['best_rank'])
        rng.setstate(saved['rng_state'])
        generator.set_state(saved['generator_state'])
        queue = saved['queue']
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=_init_worker, initargs=(dataset,)) as pool:
        if best_result is None:
            best_result = _validation(model, dataset, pool, seed)
            best_rank = (best_result['n_legal'], best_result['mean_reward'])
            save_checkpoint(out_dir / 'policy.pt', model, dict(configuration, stage='imitation',
                            update=0, validation=best_result))
            append_json(out_dir / 'validation.jsonl', {'update': 0, 'result': best_result})
            print(f'seed={seed} initial validation legal={best_result["n_legal"]}/{len(dataset["validation"])} '
                  f'score={best_result["aggregate_score"]:.6f}', flush=True)
        while update < max_updates:
            if update >= min_updates and stale_checks >= patience:
                break
            weights = state_bytes(model)
            indices = []
            while len(indices) < batch_episodes:
                if not queue:
                    queue = list(range(len(dataset['train'])))
                    rng.shuffle(queue)
                indices.append(queue.pop())
            jobs = [(weights, 'train', index, seed * 1000000 + update * batch_episodes + j, True)
                    for j, index in enumerate(indices)]
            episodes = list(pool.map(_worker_episode, jobs))
            progress = update / max(1, max_updates - 1)
            lr = 3e-4 * (1 - .9 * progress)
            for group in optimizer.param_groups:
                group['lr'] = lr
            diagnostics = ppo_update(model, optimizer, episodes, entropy=.01 + (.001 - .01) * progress,
                                     generator=generator)
            update += 1
            rows = [episode['row'] for episode in episodes]
            log = {'update': update, 'episodes': update * batch_episodes,
                   'n_legal': sum(row['legal'] for row in rows),
                   'mean_reward': sum(row['reward'] for row in rows) / len(rows),
                   'elapsed_s': time.perf_counter() - started, 'learning_rate': lr,
                   'diagnostics': diagnostics, 'cases': rows}
            append_json(out_dir / 'updates.jsonl', log)
            del episodes
            print(f'seed={seed} update={update}: legal={log["n_legal"]}/{batch_episodes}, '
                  f'reward={log["mean_reward"]:.5f}, decisions={diagnostics["decisions"]}, '
                  f'elapsed={log["elapsed_s"]:.1f}s', flush=True)
            if update % validation_interval == 0 or update == max_updates:
                result = _validation(model, dataset, pool, seed)
                rank = (result['n_legal'], result['mean_reward'])
                save_checkpoint(out_dir / 'checkpoints' / f'update_{update:04d}.pt', model,
                                dict(configuration, stage='ppo', update=update, validation=result))
                append_json(out_dir / 'validation.jsonl', {'update': update, 'result': result})
                if rank > best_rank:
                    best_rank, best_result, best_update, stale_checks = rank, result, update, 0
                    save_checkpoint(out_dir / 'policy.pt', model,
                                    dict(configuration, stage='ppo', update=update, validation=result))
                else:
                    stale_checks += 1
                print(f'seed={seed} validation update={update}: legal={result["n_legal"]}/{len(dataset["validation"])}, '
                      f'score={result["aggregate_score"]:.6f}, best_update={best_update}, stale={stale_checks}', flush=True)
            snapshot = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'update': update,
                        'stale_checks': stale_checks, 'best_update': best_update, 'best_result': best_result,
                        'best_rank': best_rank, 'rng_state': rng.getstate(), 'generator_state': generator.get_state(),
                        'queue': queue}
            temporary = latest_path.with_suffix('.tmp')
            torch.save(snapshot, temporary)
            temporary.replace(latest_path)
    result = {'seed': seed, 'updates': update, 'episodes': update * batch_episodes,
              'best_update': best_update, 'validation': best_result,
              'elapsed_this_session_s': time.perf_counter() - started,
              'stop_reason': 'validation_plateau' if update < max_updates else 'max_updates',
              'configuration': configuration}
    write_json(out_dir / 'training.json', result)
    return result


def train_all(data_dir, out_dir, suite='benchmarks_hard', seeds=(42, 43, 44), workers=4, **kwargs):
    dataset = load_dataset(data_dir, suite)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        result = train_seed(data_dir, dataset, out_dir / f'seed_{seed}', seed, workers=workers, **kwargs)
        results.append(result)
    best = max(results, key=lambda r: (r['validation']['n_legal'], r['validation']['mean_reward'], -r['seed']))
    shutil.copy2(out_dir / f'seed_{best["seed"]}' / 'policy.pt', out_dir / 'policy.pt')
    write_json(out_dir / 'training.json', {'selected_seed': best['seed'], 'runs': results})
    return out_dir / 'policy.pt'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'train', 'run'))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--suite', default='benchmarks_hard')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--max-updates', type=int, default=200)
    parser.add_argument('--min-updates', type=int, default=50)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--validation-interval', type=int, default=10)
    parser.add_argument('--batch-episodes', type=int, default=16)
    parser.add_argument('--imitation-epochs', type=int, default=5)
    args = parser.parse_args(argv)
    for field in ('workers', 'max_updates', 'patience', 'validation_interval', 'batch_episodes', 'imitation_epochs'):
        if getattr(args, field) < 1:
            parser.error(f'--{field.replace("_", "-")} must be positive')
    root = Path(args.out_dir)
    data = root / 'data'
    if args.command in ('prepare', 'run'):
        prepare_data(data, args.suite, workers=args.workers)
    if args.command == 'prepare':
        return 0
    policy = train_all(data, root / 'training', args.suite, args.seeds, args.workers,
                       max_updates=args.max_updates, min_updates=args.min_updates, patience=args.patience,
                       validation_interval=args.validation_interval, batch_episodes=args.batch_episodes,
                       imitation_epochs=args.imitation_epochs)
    if args.command == 'run':
        evaluate_suite(args.suite, policy, root / 'evaluation')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
