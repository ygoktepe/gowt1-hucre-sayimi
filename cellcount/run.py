"""Run selection, development, blind prediction and independent evaluation."""

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean
import subprocess
import sys
from time import perf_counter

import cv2
import numpy as np

from .data import DEFAULT_SEED, Dataset, read_selection, select_frames, write_selection
from .pipeline import BASELINE, Config, build_experiments, process


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def reference_count(labels):
    return int(np.count_nonzero(np.unique(labels) > 0))


def benchmark(image, config, repeats):
    counts, times = [], []
    for _ in range(repeats):
        tick = perf_counter()
        counts.append(process(image, config).count)
        times.append((perf_counter() - tick) * 1000)
    if len(set(counts)) != 1:
        raise RuntimeError(f'Tekrarlarda farklı sayım: {config.id}')
    return counts[0], fmean(times)


def metrics(errors, times):
    return {
        'mae': fmean(abs(error) for error in errors),
        'rmse': fmean(error ** 2 for error in errors) ** 0.5,
        'bias': fmean(errors),
        'exact_accuracy': fmean(int(error == 0) for error in errors),
        'mean_ms': fmean(times),
    }


def measured_row(row, config, actual, predicted, elapsed):
    error = predicted - actual
    return dict(experiment=config.id, frame=row['frame'], filename=row['filename'],
                reference=actual, prediction=predicted, error=error, abs_error=abs(error),
                elapsed_ms=elapsed)


def complexity(config):
    return (int(config.intensity != 'original') + int(config.filter != 'none')
            + len(config.morphology) + int(config.fill_holes) + int(config.split)
            + int(config.min_area > 0))


def evaluate_configs(data, rows, configs, repeats):
    by_config = {config.id: [] for config in configs}
    # Keep TIFF I/O and reference extraction outside the stopwatch.
    for index, row in enumerate(rows):
        image = data.image(row)
        actual = reference_count(data.tracking_labels(row))
        offset = index % len(configs)
        for config in configs[offset:] + configs[:offset]:
            prediction, elapsed = benchmark(image, config, repeats)
            by_config[config.id].append(measured_row(row, config, actual, prediction, elapsed))
    return by_config


def summarized(config, cases):
    result = metrics([case['error'] for case in cases], [case['elapsed_ms'] for case in cases])
    return dict(experiment=config.id, family=config.family, intensity=config.intensity,
                filter=config.filter, segmentation=config.segmentation,
                morphology='>'.join(config.morphology) or 'none',
                parameters=json.dumps(asdict(config), ensure_ascii=False, sort_keys=True),
                active_options=complexity(config), n=len(cases), **result)


def selection_from_disk(data, out):
    selection = read_selection(out / 'selected_frames.csv')
    if not selection or selection != select_frames(data, seed=selection[0]['seed']):
        raise RuntimeError('selected_frames.csv veri arşivi ve kayıtlı tohumla eşleşmiyor.')
    return selection


def develop(data, out):
    if (out / 'final_pipeline.json').exists() or (out / 'test_predictions_locked.csv').exists():
        raise RuntimeError('Kilitli çalışma klasörünü yeniden ayarlamayın. Yeni deneme klasörü kullanın.')
    selection = selection_from_disk(data, out)
    dev = [row for row in selection if row['group'] == 'development']
    val = [row for row in selection if row['group'] == 'validation']
    configurations = [BASELINE, *build_experiments()]
    if len({config.id for config in configurations}) != len(configurations):
        raise ValueError('Deney ID değerleri benzersiz olmalıdır.')

    process(data.image(dev[0]), BASELINE)  # Warm up one OpenCV call before timing.
    trials = evaluate_configs(data, dev, configurations, repeats=1)
    summaries = [summarized(config, trials[config.id]) for config in configurations]
    write_csv(out / 'experiments.csv', summaries)
    write_csv(out / 'development_frame_results.csv', [row for cases in trials.values() for row in cases])

    candidates = []
    for family in ('fixed', 'otsu', 'adaptive'):
        eligible = [row for row in summaries if row['family'] == family and row['experiment'] != BASELINE.id]
        best = min(eligible, key=lambda row: (row['mae'], row['rmse'], row['active_options'], row['experiment']))
        candidates.append(Config(**json.loads(best['parameters'])))

    validation = evaluate_configs(data, val, [BASELINE, *candidates], repeats=3)
    ranking = [summarized(config, validation[config.id]) for config in candidates]
    ranking.sort(key=lambda row: (row['mae'], row['rmse'], row['mean_ms'],
                                 row['active_options'], row['experiment']))
    for rank, row in enumerate(ranking, 1):
        row['rank'] = rank
        row['selected'] = rank == 1
    write_csv(out / 'validation_results.csv', ranking)
    write_csv(out / 'validation_baseline.csv', [summarized(BASELINE, validation[BASELINE.id])])
    write_csv(out / 'validation_frame_results.csv', [row for cases in validation.values() for row in cases])

    statistics_rows = []
    for row in [*dev, *val]:
        image = data.image(row)
        values = np.percentile(image, [1, 50, 99, 99.9])
        statistics_rows.append(dict(group=row['group'], frame=row['frame'], dtype=str(image.dtype),
                                    width=image.shape[1], height=image.shape[0], minimum=int(image.min()),
                                    maximum=int(image.max()), p01=values[0], p50=values[1],
                                    p99=values[2], p999=values[3],
                                    reference=reference_count(data.tracking_labels(row))))
    write_csv(out / 'intensity_statistics.csv', statistics_rows)
    print(f"{len(configurations)} deney, {len(ranking)} doğrulama adayı. Seçilen yöntem {ranking[0]['experiment']}.")


def digest(path):
    return sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def lock(data, out):
    manifest_path = out / 'final_pipeline.json'
    prediction_path = out / 'test_predictions_locked.csv'
    if manifest_path.exists() or prediction_path.exists():
        raise FileExistsError('Kilit çıktısı zaten var. Tekrar çalıştırmak için ayrı bir klasör kullanın.')
    selection = selection_from_disk(data, out)
    test = [row for row in selection if row['group'] == 'test']
    winner = next(row for row in read_csv(out / 'validation_results.csv') if row['selected'] == 'True')
    final = Config(**json.loads(winner['parameters']))
    predictions = []
    for index, row in enumerate(test):
        image = data.image(row)
        ordered = (BASELINE, final) if index % 2 == 0 else (final, BASELINE)
        if index == 0:
            for config in ordered:
                process(image, config)
        results = {config.id: benchmark(image, config, repeats=3) for config in ordered}
        base_count, base_time = results[BASELINE.id]
        final_count, final_time = results[final.id]
        predictions.append(dict(**row, baseline=base_count, final=final_count,
                                baseline_ms=base_time, final_ms=final_time))

    write_csv(prediction_path, predictions)
    manifest = {
        'seed': selection[0]['seed'],
        'final': asdict(final),
        'baseline': asdict(BASELINE),
        'validation': {name: float(winner[name]) for name in ('mae', 'rmse', 'mean_ms')},
        'sha256': {path.name: digest(path) for path in
                   (out / 'selected_frames.csv', prediction_path)},
        'code_sha256': {name: digest(Path(__file__).with_name(name))
                        for name in ('data.py', 'pipeline.py', 'run.py')},
        'timing_repeats': 3,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('15 kare için iki yöntemin tahminleri kaydedildi. Bu aşama hiçbir test etiketi açmaz.')


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True,
                          timeout=55).stdout


def verify_remote_lock(repo, out, commit, branch):
    if not commit:
        raise ValueError('GitHub’daki tahmin commit kimliğini --lock-commit ile belirtin.')
    if repo.resolve() != out.resolve():
        raise ValueError('Bağımsız değerlendirmede kilit dosyaları GitHub deposunun kökünde olmalıdır.')
    origin = git(repo, 'remote', 'get-url', 'origin').decode().strip()
    if not origin.startswith(('https://github.com/', 'git@github.com:', 'ssh://git@github.com/')):
        raise ValueError('origin uzak deposu bir GitHub deposu olmalıdır.')
    lock_hash = git(repo, 'rev-parse', '--verify', f'{commit}^{{commit}}').decode().strip()
    ref = f'refs/heads/{branch}'
    remote_ref = git(repo, 'ls-remote', '--exit-code', 'origin', ref).decode().strip().split()
    remote_hash = remote_ref[0]
    if remote_hash != lock_hash:
        # Supports a rerun after subsequent commits which include the original lock.
        git(repo, 'merge-base', '--is-ancestor', lock_hash, remote_hash)
    names = ('selected_frames.csv', 'final_pipeline.json', 'test_predictions_locked.csv',
             'cellcount/data.py', 'cellcount/pipeline.py', 'cellcount/run.py')
    for name in names:
        recorded = git(repo, 'show', f'{lock_hash}:{name}').replace(b'\r\n', b'\n')
        current = (out / name).read_bytes().replace(b'\r\n', b'\n')
        if recorded != current:
            raise RuntimeError(f'{name} dosyası GitHub kilit commit’iyle eşleşmiyor.')
    return lock_hash


def evaluate(data, out, commit, branch):
    repo = Path(__file__).resolve().parents[1]
    lock_hash = verify_remote_lock(repo, out, commit, branch)
    manifest = json.loads((out / 'final_pipeline.json').read_text(encoding='utf-8'))
    # The digest covers UTF-8 text with LF line endings on any platform.
    if any(digest(out / name) != value for name, value in manifest['sha256'].items()):
        raise RuntimeError('Seçim veya tahmin dosyasının SHA-256 değeri değişti.')
    if any(digest(Path(__file__).with_name(name)) != value for name, value in manifest['code_sha256'].items()):
        raise RuntimeError('Kilitlenmiş Python kaynak dosyaları değişti.')

    chosen = [row for row in selection_from_disk(data, out) if row['group'] == 'test']
    predictions = read_csv(out / 'test_predictions_locked.csv')
    keys = ('group', 'sequence', 'block', 'frame', 'filename', 'seed')
    identifiers = [tuple(str(row[key]) for key in keys) for row in predictions]
    expected = [tuple(str(row[key]) for key in keys) for row in chosen]
    if len(predictions) != 15 or identifiers != expected:
        raise ValueError('Kilitli test tahminleri seçilen karelerle eşleşmiyor.')

    records = []
    for row in predictions:
        actual = reference_count(data.tracking_labels(row, allow_test=True))
        base_error, final_error = int(row['baseline']) - actual, int(row['final']) - actual
        records.append(dict(**row, reference=actual, baseline_error=base_error, final_error=final_error,
                            baseline_abs_error=abs(base_error), final_abs_error=abs(final_error)))
    write_csv(out / 'test_results.csv', records)
    summary = [dict(method=method, n=len(records), lock_commit=lock_hash, **metrics(
        [row[f'{method}_error'] for row in records], [float(row[f'{method}_ms']) for row in records]
    )) for method in ('baseline', 'final')]
    write_csv(out / 'test_metrics.csv', summary)
    print('Bağımsız test metrikleri kaydedildi.')


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('select', 'develop', 'lock', 'evaluate'))
    parser.add_argument('--data', required=True, type=Path, help='CTC eğitim arşivi veya açılmış klasör')
    parser.add_argument('--out', default=Path.cwd(), type=Path)
    parser.add_argument('--seed', default=DEFAULT_SEED, type=int)
    parser.add_argument('--lock-commit')
    parser.add_argument('--branch', default='main')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    with Dataset(args.data) as data:
        if args.stage == 'select':
            selection = select_frames(data, args.seed)
            destination = args.out / 'selected_frames.csv'
            if destination.exists() and read_selection(destination) != selection:
                raise FileExistsError('Mevcut kare seçiminin üzerine yazılmaz. Yeni çalışma klasörü kullanın.')
            write_selection(destination, selection)
        elif args.stage == 'develop':
            develop(data, args.out)
        elif args.stage == 'lock':
            lock(data, args.out)
        else:
            evaluate(data, args.out, args.lock_commit, args.branch)


if __name__ == '__main__':
    main()
