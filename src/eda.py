"""Summarize the complete GFP assay and render its six EDA figures."""

import argparse
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter

from data_processing import AMINO_ACIDS, ASSAY_FILENAME, file_sha256, read_csv, write_csv, write_json

BLUE = "#266794"
DARK = "#1e3343"
GRAY = "#d9dde1"
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def summarize(values):
    return {
        "count": len(values),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std_sample": float(values.std(ddof=1)) if len(values) > 1 else None,
        "quantiles": {str(p): float(v) for p, v in zip(QUANTILES, np.quantile(values, QUANTILES))},
    }


def describe_order(values):
    positions = np.arange(1, len(values) + 1)
    _, inverse, frequencies = np.unique(values, return_inverse=True, return_counts=True)
    average_ranks = frequencies.cumsum() - (frequencies - 1) / 2
    steps = np.diff(values)
    ascending, descending = bool((steps >= 0).all()), bool((steps <= 0).all())
    variable = len(frequencies) > 1
    return {
        "nondecreasing": ascending,
        "nonincreasing": descending,
        "adjacent_increases": int((steps > 0).sum()),
        "adjacent_decreases": int((steps < 0).sum()),
        "adjacent_ties": int((steps == 0).sum()),
        "pearson_row_index_vs_fitness": float(np.corrcoef(positions, values)[0, 1]) if variable else None,
        "spearman_row_index_vs_fitness": (
            float(np.corrcoef(positions, average_ranks[inverse])[0, 1]) if variable else None
        ),
        "consecutive_row_quartiles": [
            {"first_data_row": int(block[0]), "last_data_row": int(block[-1]),
             "median_fitness": float(np.median(values[block - 1]))}
            for block in np.array_split(positions, 4) if len(block)
        ],
    }


def calculate_tables(variants, changes, wt, cutoff, output):
    count_by_position = Counter(int(row["position"]) for row in changes)
    substitutions = defaultdict(set)
    for row in changes:
        substitutions[int(row["position"])].add(row["mutant_aa"])
    positions = [{
        "position": p, "wt_aa": aa, "variant_count": count_by_position[p],
        "variant_fraction": count_by_position[p] / len(variants),
        "unique_substitution_count": len(substitutions[p]),
    } for p, aa in enumerate(wt, start=1)]
    write_csv(output / "position_mutation_frequency.csv", tuple(positions[0]), positions)

    scores = np.array([float(row["fitness"]) for row in variants])
    by_count = defaultdict(list)
    for row in variants:
        by_count[int(row["mutation_count"])].append(row)
    bin_counts = Counter(int(row["fitness_bin"]) for row in variants)
    groups = {n: np.array([float(row["fitness"]) for row in rows]) for n, rows in sorted(by_count.items())}
    grouped_stats = []
    for mutation_count, values in groups.items():
        count_at_cutoff = sum(Decimal(row["fitness"]) >= cutoff for row in by_count[mutation_count])
        grouped_stats.append({
            "mutation_count": mutation_count,
            "variant_count": len(values),
            "variant_fraction": len(values) / len(variants),
            "mean_fitness": float(values.mean()),
            "median_fitness": float(np.median(values)),
            "min_fitness": float(values.min()),
            "max_fitness": float(values.max()),
            "at_or_above_threshold_count": count_at_cutoff,
            "at_or_above_threshold_fraction": count_at_cutoff / len(values),
        })
    write_csv(output / "fitness_by_mutation_count.csv", tuple(grouped_stats[0]), grouped_stats)

    singles = by_count[1]
    heatmap = np.full((len(AMINO_ACIDS), len(wt)), np.nan)
    cells = {}
    for row in singles:
        p, alt = int(row["mutation_positions"]), row["mutant_amino_acids"]
        if (p, alt) in cells:
            raise ValueError(f"Duplicate independently measured single substitution at {p} to {alt}")
        cells[p, alt] = row["fitness"]
        heatmap[AMINO_ACIDS.index(alt), p - 1] = float(row["fitness"])
    heatmap_records = [
        {"mutant_aa": alt, **{str(p): cells.get((p, alt), "NA") for p in range(1, len(wt) + 1)}}
        for alt in AMINO_ACIDS
    ]
    write_csv(output / "single_substitution_heatmap.csv", tuple(heatmap_records[0]), heatmap_records)

    covered_positions = sum(bool(count) for count in count_by_position.values())
    single_positions = len({p for p, _ in cells})
    summary = {
        "sample_count": len(variants),
        "fitness": summarize(scores),
        "fitness_threshold": {"cutoff": float(cutoff), "comparison": ">="},
        "fitness_bin": {
            str(label): {"count": bin_counts[label], "fraction": bin_counts[label] / len(variants)}
            for label in (0, 1)
        },
        "mutation_count_range": [min(groups), max(groups)],
        "candidate_counts": {
            "exactly_1_to_4": {str(n): len(by_count[n]) for n in range(1, 5)},
            "between_1_and_4_inclusive": sum(len(by_count[n]) for n in range(1, 5)),
            "all_mutation_counts": len(variants),
        },
        "coverage": {
            "unique_substitution_count_all_variants": sum(map(len, substitutions.values())),
            "positions_covered_all_variants": covered_positions,
            "position_fraction_all_variants": covered_positions / len(wt),
            "positions_covered_by_independent_singles": single_positions,
            "position_fraction_independent_singles": single_positions / len(wt),
            "reference_length": len(wt),
        },
        "independently_measured_singles": {
            "fitness": summarize(groups[1]),
            "heatmap_cells_measured": len(cells),
            "heatmap_cells_na": int(np.isnan(heatmap).sum()),
            "unmeasured_possible_substitution_cells": len(wt) * (len(AMINO_ACIDS) - 1) - len(cells),
        },
        "original_fitness_row_order": describe_order(scores),
    }
    write_json(output / "eda_summary.json", summary)
    return scores, groups, grouped_stats, positions, heatmap


def make_figures(output, scores, groups, grouped_stats, positions, heatmap, cutoff):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13,
        "axes.titleweight": "bold", "axes.titlepad": 11, "axes.labelcolor": DARK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "text.color": DARK, "xtick.color": DARK, "ytick.color": DARK,
        "grid.color": "#e4ebef", "grid.linewidth": 0.65,
        "savefig.dpi": 300, "savefig.facecolor": "white",
    })

    def new_figure(title, xlabel, ylabel, size=(7.6, 4.2), grid=True):
        fig, ax = plt.subplots(figsize=size, layout="constrained")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        if grid:
            ax.grid(axis="y")
        return fig, ax

    def save(fig, name):
        fig.savefig(output / f"{name}.png")
        plt.close(fig)

    fig, ax = new_figure("Distribution of GFP DMS scores", "DMS score", "Variant count")
    ax.hist(scores, bins=40, color=BLUE, edgecolor="white", linewidth=0.45)
    ax.axvline(float(cutoff), color="#a44931", linestyle="--", linewidth=1.3,
               label=f"ProteinGym cutoff ({cutoff})")
    ax.legend(frameon=False)
    save(fig, "figure_1_dms_score_distribution")

    counts = list(groups)
    fig, ax = new_figure("Distribution of GFP mutation counts", "Substitutions per variant", "Variant count")
    ax.bar(counts, [len(values) for values in groups.values()], color=BLUE, width=0.76)
    ax.set_xticks(counts)
    save(fig, "figure_2_mutation_count_distribution")

    fig, ax = new_figure("Fitness by GFP mutation count", "Substitutions per variant", "DMS score")
    ax.boxplot(list(groups.values()), positions=counts, widths=0.62, patch_artist=True,
               boxprops={"facecolor": "#d7e8f3", "edgecolor": BLUE},
               medianprops={"color": DARK, "linewidth": 1.35},
               flierprops={"marker": ".", "markersize": 2.2, "markerfacecolor": BLUE,
                           "markeredgecolor": BLUE, "alpha": 0.20})
    ax.set_xticks(counts)
    save(fig, "figure_3_fitness_by_mutation_count")

    fig, ax = new_figure("Variants meeting the ProteinGym fitness threshold",
                         "Substitutions per variant", f"Variants with DMS score ≥ {cutoff}")
    fractions = [row["at_or_above_threshold_fraction"] for row in grouped_stats]
    ax.bar(counts, fractions, width=0.76, color=BLUE)
    ax.set_xticks(counts)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    save(fig, "figure_4_fitness_threshold_by_mutation_count")

    fig, ax = new_figure("Mutation frequency across the GFP sequence",
                         "Reference sequence position", "Fraction of variants mutated", size=(10, 3.9))
    ax.bar([row["position"] for row in positions], [row["variant_fraction"] for row in positions],
           width=0.9, color=BLUE, linewidth=0)
    ax.set_xlim(0.5, len(positions) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig, "figure_5_position_mutation_frequency")

    fig, ax = new_figure("Independently measured GFP single-substitution fitness",
                         "Reference sequence position", "Substituted amino acid", size=(14, 5.0), grid=False)
    colors = plt.get_cmap("viridis").with_extremes(bad=GRAY)
    image = ax.imshow(np.ma.masked_invalid(heatmap), interpolation="nearest", aspect="auto", cmap=colors,
                      extent=(0.5, heatmap.shape[1] + 0.5, len(AMINO_ACIDS) - 0.5, -0.5))
    ax.set_yticks(range(len(AMINO_ACIDS)), list(AMINO_ACIDS))
    ax.set_xticks([1, *range(20, heatmap.shape[1] - 7, 20), heatmap.shape[1]])
    fig.colorbar(image, ax=ax, label="Experimental DMS score", pad=0.015, fraction=0.025)
    ax.text(0.995, 1.015, "Gray = NA", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    save(fig, "figure_6_single_substitution_heatmap")


def run(root):
    processed = root / "data" / "processed"
    output = root / "results" / "eda"
    quality = json.loads((output / "data_quality_report.json").read_text(encoding="utf-8"))
    if quality["status"] != "passed":
        raise ValueError("Run data_processing.py successfully before EDA.")
    for name, digest in quality["source_files_sha256"].items():
        if file_sha256(root / "data" / "raw" / name) != digest:
            raise ValueError(f"Input {name} changed since processing. Rerun data_processing.py.")
    metadata = json.loads((processed / "gfp_metadata.json").read_text(encoding="utf-8"))
    wt = "".join((processed / "gfp_wildtype.fasta").read_text(encoding="utf-8").splitlines()[1:])
    _, variant_rows = read_csv(processed / "gfp_processed.csv")
    _, mutation_rows = read_csv(processed / "gfp_mutations_long.csv")
    _, raw_rows = read_csv(root / "data" / "raw" / ASSAY_FILENAME)
    variants = [row for _, row in variant_rows]
    changes = [row for _, row in mutation_rows]
    if (len(wt) != metadata["sequence_length"]
            or len(variants) != quality["processed_variant_count"]
            or len(changes) != quality["mutation_long_row_count"]):
        raise ValueError("Processed file counts disagree with the quality report or reference metadata.")
    if [(row["mutant"], row["fitness"]) for row in variants] != [
        (row["mutant"], row["DMS_score"]) for _, row in raw_rows
    ]:
        raise ValueError("Processed variants must retain the source fitness values and row order.")
    cutoff = Decimal(metadata["binarization_cutoff_source_value"])
    tables = calculate_tables(variants, changes, wt, cutoff, output)
    make_figures(output, *tables, cutoff)
    print(f"Wrote EDA tables and six 300 DPI figures to {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    run(parser.parse_args().root.resolve())
