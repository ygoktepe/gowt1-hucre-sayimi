"""Render report evidence into a local directory outside the repository."""

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .data import Dataset, read_selection
from .pipeline import BASELINE, Config, build_experiments, process
from .run import read_csv, reference_count, verify_remote_lock


INK = '#132537'
BLUE = '#254d79'
TEAL = '#087a72'
OCHRE = '#a4532d'
GREY = '#45576a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.titlesize': 10,
                     'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': '#bccee0'})


def finish(figure, output, name):
    figure.savefig(output / f'{name}.png', dpi=170, facecolor='white', bbox_inches='tight', pad_inches=.12)
    plt.close(figure)


def show(ax, array, title, *, mask=False, vmax=None):
    limit = 1 if mask else (vmax if vmax is not None else max(1, np.percentile(array, 99.7)))
    ax.imshow(array, cmap='gray', vmin=0, vmax=limit, interpolation='nearest')
    ax.set_title(title, loc='left', fontweight='semibold', pad=6)
    ax.set_axis_off()


def overlay(ax, image, labels, title, tracking=None):
    show(ax, image, title)
    if np.any(labels):
        ax.contour(labels > 0, levels=[.5], colors=['#58e8d4'], linewidths=.50)
        # A boundary between two foreground IDs is otherwise invisible in a binary contour.
        interior_edge = (labels[:-1] > 0) & (labels[1:] > 0) & (labels[:-1] != labels[1:])
        yy, xx = np.where(interior_edge)
        ax.scatter(xx, yy, s=.10, c='#58e8d4', linewidths=0)
    if tracking is not None:
        for label in np.unique(tracking[tracking > 0]):
            yy, xx = np.where(tracking == label)
            ax.scatter(xx.mean(), yy.mean(), marker='+', s=15, c='#ffcf64', linewidths=.8)


def intensity_figure(image, configs, output):
    pairs = [('E02', 'Özgün'), ('E04', 'Germe (%1–99,5)'),
             ('E05', 'Histogram eşitleme'), ('E06', 'CLAHE (2,0 / 8×8)'),
             ('E07', 'Gamma (0,40)'), ('E28', 'Gamma (0,25)')]
    fig, axes = plt.subplots(2, len(pairs), figsize=(14.0, 4.05), layout='constrained',
                             gridspec_kw={'height_ratios': [3.2, 1]})
    for column, (key, name) in enumerate(pairs):
        intensity = process(image, configs[key], keep_steps=True).steps['intensity']
        show(axes[0, column], intensity, name, vmax=255)
        counts, edges = np.histogram(intensity, bins=256, range=(0, 256))
        histogram = axes[1, column]
        histogram.plot(edges[:-1], counts, lw=.8, color=TEAL)
        histogram.set(xlim=(0, 255), ylim=(1, 2_000_000), yscale='log', xticks=[0, 128, 255])
        histogram.set_yticks([1, 1000, 1000000] if column == 0 else [])
        histogram.set_xlabel('Yoğunluk')
    axes[1, 0].set_ylabel('Piksel (log)')
    finish(fig, output, 'fig02_intensity')


def comparison_figure(image, configs, output):
    display = [('E01', 'Sabit eşik (T=9)', 'threshold'), ('E02', 'Otsu', 'threshold'),
               ('E03', 'Adaptif eşik (121, C=−4)', 'threshold'), ('E13', 'Erozyon (5×5)', 'morphology'),
               ('E14', 'Genişletme (5×5)', 'morphology'), ('E01', 'Açma ve kapama (5×5)', 'morphology')]
    fig, axes = plt.subplots(2, 3, figsize=(10.1, 5.85), layout='constrained')
    for ax, (key, name, step) in zip(axes.flat, display):
        result = process(image, configs[key], keep_steps=True)
        show(ax, result.steps[step], name, mask=True)
    finish(fig, output, 'fig03_comparison')


def pipeline_figure(image, config, output):
    result = process(image, config, keep_steps=True)
    panels = [('original', 'Ham kare'), ('intensity', 'Yoğunluk adımı'),
              ('filtered', 'Uzamsal filtre'), ('threshold', 'Eşik maskesi'),
              ('morphology', 'Morfoloji ve delik doldurma')]
    fig, axes = plt.subplots(2, 3, figsize=(10.1, 5.85), layout='constrained')
    for ax, (key, title) in zip(axes.flat, panels):
        show(ax, result.steps[key], title, mask=key in ('threshold', 'morphology'))
    overlay(axes.flat[-1], image, result.labels, f'Etiketler ({result.count} adet)')
    finish(fig, output, 'fig04_steps')


def temporal_figure(rows, output):
    scored = 'reference' in rows[0]
    fig, axes = plt.subplots(1, 3, figsize=(11.15, 3.15), layout='constrained')
    series = [('final', 'Nihai yöntem', TEAL, 's', '-'),
              ('baseline', 'Baseline', OCHRE, '^', '--')]
    if scored:
        series.insert(0, ('reference', 'TRA referansı', BLUE, 'o', ':'))
    baseline_max = max(int(row['baseline']) for row in rows)
    other_max = max(int(row[key]) for row in rows for key in ('final', 'reference') if key in row)
    separate_axis = baseline_max > 3 * max(other_max, 1)
    foreground = [int(row[key]) for row in rows for key in ('final', 'reference') if key in row]
    base_values = [int(row['baseline']) for row in rows]
    for block, ax in enumerate(axes, 1):
        subset = [row for row in rows if int(row['block']) == block]
        x = [int(row['frame']) for row in subset]
        other_ax = ax.twinx() if separate_axis else ax
        handles = []
        for key, name, color, marker, style in series:
            axis = other_ax if key == 'baseline' else ax
            line, = axis.plot(x, [int(row[key]) for row in subset], label=name, color=color,
                              marker=marker, markersize=4, lw=1.45, linestyle=style)
            handles.append(line)
        ax.set(title=f'Blok {block} · t{x[0]:03d}–t{x[-1]:03d}', xlabel='Kare', xticks=x)
        ax.grid(axis='y', alpha=.2)
        if separate_axis:
            ax.set_ylim(min(foreground) - 2, max(foreground) + 2)
            other_ax.set_ylim(min(base_values) - 30, max(base_values) + 30)
            other_ax.tick_params(axis='y', labelcolor=OCHRE, labelright=block == 3)
    axes[0].set_ylabel('Referans ve nihai sayım' if scored and separate_axis else 'Hücre sayısı')
    if separate_axis:
        other_ax.set_ylabel('Baseline sayısı', color=OCHRE)
    axes[1].legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -0.23),
                   ncol=len(series), frameon=False, fontsize=8)
    finish(fig, output, 'fig05_temporal')


def marker_diagnostics(labels, tracking):
    linked = Counter()
    missing = 0
    fragmented = 0
    for label in np.unique(tracking[tracking > 0]):
        predicted = np.unique(labels[tracking == label])
        positive = predicted[predicted > 0]
        missing += len(positive) == 0
        fragmented += len(positive) > 1
        linked.update(positive.tolist())
    return dict(uncovered_reference_ids=int(missing), multiply_overlapping_reference_ids=int(fragmented),
                labels_overlapping_multiple_reference_ids=sum(value > 1 for value in linked.values()),
                unlinked_predicted_labels=int(labels.max()) - len(linked))


def examples_figure(data, samples, config, output, *, tested):
    ranked = sorted(samples, key=lambda pair: (abs(pair[2] - pair[1]), pair[0]['frame']))
    easy, hardest = ranked[0], ranked[-1]
    remaining = [pair for pair in samples if pair not in (easy, hardest)]
    if tested:
        ordered = sorted(samples, key=lambda pair: pair[0]['frame'])
        jumps = [(abs((right[2] - right[1]) - (left[2] - left[1])), right[0]['frame'], right)
                 for left, right in zip(ordered, ordered[1:])
                 if left[0]['block'] == right[0]['block'] and right in remaining]
        hard = max(jumps)[2]
    else:
        hard = max(remaining, key=lambda pair: (abs(pair[2] - pair[1]), pair[0]['frame']))
    fig, axes = plt.subplots(1, 3, figsize=(10.35, 3.45), layout='constrained')
    diagnostics = []
    middle = 'Zamansal sıçrama' if tested else 'Diğer örnek'
    for ax, name, (row, reference, prediction) in zip(axes, ('En düşük hata', middle, 'En yüksek hata'),
                                                     (easy, hard, hardest)):
        image = data.image(row)
        truth = data.tracking_labels(row, allow_test=tested)
        result = process(image, config)
        assert result.count == prediction
        show_title = f"{name} · t{row['frame']:03d}\nReferans {reference} / Tahmin {prediction}"
        overlay(ax, image, result.labels, show_title, truth)
        diagnostics.append(dict(frame=row['frame'], reference=reference, prediction=prediction,
                                **marker_diagnostics(result.labels, truth)))
    finish(fig, output, 'fig06_examples')
    (output / 'examples.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')


def describe_test_images(data, rows, output):
    statistics = []
    for row in rows:
        image = data.image(row)
        statistics.append(dict(frame=row['frame'], shape=image.shape, dtype=str(image.dtype),
                               median=float(np.median(image)), p99=float(np.percentile(image, 99)),
                               maximum=int(image.max())))
    (output / 'test_intensity_statistics.json').write_text(json.dumps(statistics, indent=2), encoding='utf-8')


def contact_sheet(data, rows, config, output):
    fig, axes = plt.subplots(2, 5, figsize=(11.3, 5.15), layout='constrained')
    for ax, row in zip(axes.flat, rows):
        image = data.image(row)
        truth = data.tracking_labels(row)
        result = process(image, config)
        overlay(ax, image, result.labels, f"t{row['frame']:03d} · {reference_count(truth)}/{result.count}", truth)
    finish(fig, output, 'appendix_development')


def baseline_figure(image, output):
    result = process(image, BASELINE, keep_steps=True)
    crop = np.s_[288:672, 288:672]
    fig, axes = plt.subplots(1, 3, figsize=(10.35, 3.5), layout='constrained')
    show(axes[0], image[crop], 'Ham TIFF · merkez kesiti')
    show(axes[1], result.steps['threshold'][crop], 'Otsu ön planı', mask=True)
    overlay(axes[2], image[crop], result.labels[crop], 'B00 bileşenlerinin konturları')
    finish(fig, output, 'appendix_baseline')


def watershed_figure(image, configs, output):
    before = process(image, configs['E03'])
    after = process(image, configs['E27'], keep_steps=True)
    parents = np.unique(before.labels[after.steps['markers'] > 0])
    for parent in parents[parents > 0]:
        region = before.labels == parent
        if np.count_nonzero(np.unique(after.labels[region]) > 0) > 1:
            yy, xx = np.where(region)
            crop = np.s_[max(0, yy.min() - 34):yy.max() + 35, max(0, xx.min() - 34):xx.max() + 35]
            break
    else:
        crop = np.s_[:, :]
    fig, axes = plt.subplots(1, 3, figsize=(10.35, 3.45), layout='constrained')
    overlay(axes[0], image[crop], before.labels[crop], f'E03 · karede {before.count} etiket')
    show(axes[1], after.steps['distance'][crop], 'Öklid uzaklığı ve işaretleyiciler')
    yy, xx = np.where(after.steps['markers'][crop] > 0)
    axes[1].scatter(xx, yy, marker='+', s=30, c='#ffcf64', linewidths=1.2)
    overlay(axes[2], image[crop], after.labels[crop], f'E27 · karede {after.count} etiket')
    finish(fig, output, 'appendix_watershed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--results', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock-commit', help='GitHub commit kimliği; test örneği çizmek için gerekir')
    parser.add_argument('--branch', default='main')
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError('Ders görsellerini depo dışında özel bir klasöre kaydedin.')
    args.output.mkdir(parents=True, exist_ok=True)
    selected = read_selection(args.results / 'selected_frames.csv')
    config = Config(**json.loads((args.results / 'final_pipeline.json').read_text(encoding='utf-8'))['final'])
    configs = {method.id: method for method in [BASELINE, *build_experiments()]}
    dev = [row for row in selected if row['group'] == 'development']
    val = {row['frame']: row for row in selected if row['group'] == 'validation'}
    test = {row['frame']: row for row in selected if row['group'] == 'test'}
    scored = args.results / 'test_results.csv'
    tested = scored.is_file()
    if tested:
        repo = Path(__file__).resolve().parents[1]
        actual_lock = verify_remote_lock(repo, args.results, args.lock_commit, args.branch)
        test_metrics = read_csv(args.results / 'test_metrics.csv')
        if not test_metrics or any(row['lock_commit'] != actual_lock for row in test_metrics):
            raise ValueError('Test metriklerinde doğrulanmış tahmin commit’i bulunamadı.')
    scope = test if tested else val
    path = scored if tested else args.results / 'validation_frame_results.csv'
    samples = [(scope[int(row['frame'])], int(row['reference']), int(row['final' if tested else 'prediction']))
               for row in read_csv(path) if tested or row['experiment'] == config.id]
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    with Dataset(args.data) as data:
        intensity_figure(data.image(dev[5]), configs, args.output)
        comparison_figure(data.image(dev[5]), configs, args.output)
        pipeline_figure(data.image(dev[5]), config, args.output)
        examples_figure(data, samples, config, args.output, tested=tested)
        contact_sheet(data, dev, config, args.output)
        baseline_figure(data.image(dev[3]), args.output)
        watershed_figure(data.image(dev[-1]), configs, args.output)
        if tested:
            describe_test_images(data, test.values(), args.output)
    path = scored if tested else args.results / 'test_predictions_locked.csv'
    temporal_figure(read_csv(path), args.output)


if __name__ == '__main__':
    main()
