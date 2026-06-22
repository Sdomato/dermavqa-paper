"""
Genera visualizaciones livianas para el baseline textual de retrieval.

Por defecto lee `metrics_summary.csv` y, si existe, `bertscore_summary.csv`
desde `results/dataset_enriched/retrieval_textual/`. Produce graficos de barras
por modelo y split para chrF, sacreBLEU y BERTScore F1, destacando el ganador
de valid por chrF.

Uso:
    python -m src.plot_retrieval_scores
    python -m src.plot_retrieval_scores --results-dir results/dataset_enriched/retrieval_textual
    python -m src.plot_retrieval_scores --score-distributions
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SCORE_KEYS = ("similarity_score", "retrieved_score", "score")
DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "dataset_enriched"
    / "retrieval_textual"
)

METHOD_LABELS = {
    "tfidf": "TF-IDF",
    "e5_small": "E5 small",
    "sbert_multilingual_minilm": "SBERT MiniLM",
}
SPLIT_LABELS = {"valid": "Valid", "test": "Test", "train": "Train"}
SPLIT_COLORS = {"valid": "#4C72B0", "test": "#55A868", "train": "#C44E52"}
HIGHLIGHT_COLOR = "#D18F00"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def load_metric_rows(path: Path, numeric_columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_row in read_csv_rows(path):
        row: dict[str, Any] = dict(raw_row)
        for column in numeric_columns:
            if row.get(column) not in (None, ""):
                row[column] = float(row[column])
        rows.append(row)
    return rows


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def split_sort_key(split: str) -> tuple[int, str]:
    order = {"valid": 0, "test": 1, "train": 2}
    return (order.get(split, 99), split)


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


def split_label(split: str) -> str:
    return SPLIT_LABELS.get(split, split.title())


def value_label(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def valid_chrf_winner(metrics_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid_rows = [row for row in metrics_rows if row.get("split") == "valid" and "chrf" in row]
    if not valid_rows:
        return None
    return max(valid_rows, key=lambda row: float(row["chrf"]))


def plot_grouped_bars(
    rows: list[dict[str, Any]],
    metric: str,
    output_path: Path,
    ylabel: str,
    title: str,
    decimals: int,
    highlight_valid_winner: bool = False,
) -> Path | None:
    plot_rows = [row for row in rows if row.get(metric) not in (None, "")]
    if not plot_rows:
        print(f"  Sin datos para {metric}; no se genera {output_path.name}")
        return None

    methods = ordered_unique([str(row["method"]) for row in plot_rows])
    splits = sorted(ordered_unique([str(row["split"]) for row in plot_rows]), key=split_sort_key)
    values = {
        (str(row["method"]), str(row["split"])): float(row[metric])
        for row in plot_rows
    }

    fig_width = max(7.0, 1.5 * len(methods) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))

    bar_width = min(0.28, 0.75 / max(len(splits), 1))
    base_positions = list(range(len(methods)))
    max_value = max(values.values())
    winner_method = None
    winner_value = None
    if highlight_valid_winner:
        valid_rows = [row for row in plot_rows if row.get("split") == "valid"]
        if valid_rows:
            winner = max(valid_rows, key=lambda row: float(row[metric]))
            winner_method = str(winner["method"])
            winner_value = float(winner[metric])

    for split_index, split in enumerate(splits):
        offset = (split_index - (len(splits) - 1) / 2) * bar_width
        x_positions = [position + offset for position in base_positions]
        heights = [values.get((method, split), 0.0) for method in methods]

        edgecolors = [
            HIGHLIGHT_COLOR if split == "valid" and method == winner_method else "white"
            for method in methods
        ]
        linewidths = [
            2.4 if split == "valid" and method == winner_method else 0.8
            for method in methods
        ]

        bars = ax.bar(
            x_positions,
            heights,
            width=bar_width,
            label=split_label(split),
            color=SPLIT_COLORS.get(split, "#8172B2"),
            edgecolor=edgecolors,
            linewidth=linewidths,
        )

        for bar, value in zip(bars, heights):
            if value <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_value * 0.015,
                value_label(value, decimals),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    if winner_method is not None and winner_value is not None:
        winner_x = base_positions[methods.index(winner_method)]
        valid_offset = (splits.index("valid") - (len(splits) - 1) / 2) * bar_width
        ax.annotate(
            "Ganador valid por chrF",
            xy=(winner_x + valid_offset, winner_value),
            xytext=(0, 28),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=HIGHLIGHT_COLOR,
            arrowprops={"arrowstyle": "->", "color": HIGHLIGHT_COLOR, "lw": 1.2},
        )

    ax.set_xticks(base_positions)
    ax.set_xticklabels([method_label(method) for method in methods], rotation=12, ha="right")
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_title(title, pad=12)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(title="Split", frameon=True, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_ylim(0, max_value * 1.22 if max_value > 0 else 1)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {output_path}")
    return output_path


def make_metric_plots(results_dir: Path = DEFAULT_RESULTS_DIR) -> list[Path]:
    generated: list[Path] = []
    metrics_path = results_dir / "metrics_summary.csv"
    bertscore_path = results_dir / "bertscore_summary.csv"

    if metrics_path.exists():
        metrics_rows = load_metric_rows(metrics_path, ["sacrebleu", "chrf"])
        winner = valid_chrf_winner(metrics_rows)
        if winner is not None:
            print(
                "Ganador valid por chrF: "
                f"{method_label(str(winner['method']))} ({float(winner['chrf']):.3f})"
            )

        for path in [
            plot_grouped_bars(
                rows=metrics_rows,
                metric="chrf",
                output_path=results_dir / "retrieval_textual_chrf_by_model_split.png",
                ylabel="chrF",
                title="Baseline textual dataset_enriched - chrF por modelo y split",
                decimals=2,
                highlight_valid_winner=True,
            ),
            plot_grouped_bars(
                rows=metrics_rows,
                metric="sacrebleu",
                output_path=results_dir / "retrieval_textual_sacrebleu_by_model_split.png",
                ylabel="sacreBLEU",
                title="Baseline textual dataset_enriched - sacreBLEU por modelo y split",
                decimals=2,
            ),
        ]:
            if path is not None:
                generated.append(path)
    else:
        print(f"No existe {metrics_path}; se omiten chrF y sacreBLEU.")

    if bertscore_path.exists():
        bertscore_rows = load_metric_rows(
            bertscore_path,
            ["bertscore_precision", "bertscore_recall", "bertscore_f1"],
        )
        path = plot_grouped_bars(
            rows=bertscore_rows,
            metric="bertscore_f1",
            output_path=results_dir / "retrieval_textual_bertscore_f1_by_model_split.png",
            ylabel="BERTScore F1",
            title="Baseline textual dataset_enriched - BERTScore F1 por modelo y split",
            decimals=3,
        )
        if path is not None:
            generated.append(path)
    else:
        print(f"No existe {bertscore_path}; se omite BERTScore.")

    return generated


def load_scores(path: Path) -> tuple[list[float], str]:
    if path.suffix == ".json":
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            rows = raw
        elif isinstance(raw, dict):
            lists = [value for value in raw.values() if isinstance(value, list)]
            if not lists:
                raise ValueError(f"No se encontro una lista en {path}")
            rows = lists[0]
        else:
            raise ValueError(f"Formato JSON no reconocido en {path}")
    elif path.suffix == ".csv":
        rows = read_csv_rows(path)
    else:
        raise ValueError(f"Extension no soportada: {path.suffix}")

    if not rows:
        raise ValueError(f"Archivo sin filas: {path}")

    score_key = next((key for key in SCORE_KEYS if key in rows[0]), None)
    if score_key is None:
        raise KeyError(f"No se encontro ninguna columna de score en {path.name}: {SCORE_KEYS}")

    scores: list[float] = []
    for row in rows:
        value = row.get(score_key)
        if value in (None, ""):
            continue
        scores.append(float(value))
    if not scores:
        raise ValueError(f"No hay scores numericos en {path}")
    return scores, score_key


def collect_score_files(results_dir: Path) -> list[Path]:
    excluded = {
        "metrics_summary.csv",
        "bertscore_summary.csv",
        "manual_review_valid_10.csv",
    }
    files = [
        path
        for path in sorted([*results_dir.glob("*.json"), *results_dir.glob("*.csv")])
        if path.name not in excluded and not path.name.startswith(".")
    ]
    return files


def plot_score_histogram(scores: list[float], score_key: str, label: str, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = min(30, max(5, int(len(scores) ** 0.5)))
    ax.hist(
        scores,
        bins=bins,
        density=True,
        color="#4C72B0",
        alpha=0.45,
        edgecolor="white",
        linewidth=0.6,
        label="Histograma",
    )

    mean_value = mean(scores)
    median_value = median(scores)
    ax.axvline(mean_value, color="#DD4444", linestyle="--", linewidth=1.4, label=f"Media = {mean_value:.3f}")
    ax.axvline(median_value, color="#228B22", linestyle=":", linewidth=1.4, label=f"Mediana = {median_value:.3f}")

    ax.set_xlabel(score_key, labelpad=8)
    ax.set_ylabel("Densidad", labelpad=8)
    ax.set_title(f"Distribucion de scores - {label}", pad=12)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {output_path}")
    return output_path


def plot_score_boxplot(scores: list[float], score_key: str, label: str, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.boxplot(
        scores,
        vert=True,
        patch_artist=True,
        widths=0.45,
        medianprops={"color": "#DD4444", "linewidth": 2},
        boxprops={"facecolor": "#4C72B0", "alpha": 0.55},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4, "markerfacecolor": "#4C72B0"},
    )

    median_value = median(scores)
    ax.text(
        1.08,
        median_value,
        f"Mediana\n{median_value:.3f}",
        va="center",
        ha="left",
        fontsize=8,
        color="#DD4444",
        transform=ax.get_yaxis_transform(),
    )

    ax.set_xticks([])
    ax.set_ylabel(score_key, labelpad=8)
    ax.set_title(f"Boxplot de scores - {label}", pad=12)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {output_path}")
    return output_path


def print_stats(scores: list[float], label: str) -> None:
    print(
        f"\n[{label}] n={len(scores)}"
        f"  media={mean(scores):.4f}"
        f"  mediana={median(scores):.4f}"
        f"  std={pstdev(scores):.4f}"
        f"  min={min(scores):.4f}"
        f"  max={max(scores):.4f}"
    )


def make_score_distribution_plots(results_dir: Path) -> list[Path]:
    generated: list[Path] = []
    files = collect_score_files(results_dir)
    if not files:
        print(f"No se encontraron archivos JSON/CSV en {results_dir}")
        return generated

    print(f"Procesando distribuciones de scores en {len(files)} archivo(s).\n")
    for path in files:
        try:
            scores, score_key = load_scores(path)
        except (KeyError, ValueError) as exc:
            print(f"  Saltando {path.name}: {exc}")
            continue

        label = path.stem
        print_stats(scores, label)
        generated.append(
            plot_score_histogram(
                scores,
                score_key=score_key,
                label=label,
                output_path=results_dir / f"{label}_hist.png",
            )
        )
        generated.append(
            plot_score_boxplot(
                scores,
                score_key=score_key,
                label=label,
                output_path=results_dir / f"{label}_boxplot.png",
            )
        )

    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera graficos agregados del baseline textual de retrieval."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Carpeta con metrics_summary.csv y bertscore_summary.csv.",
    )
    parser.add_argument(
        "--score-distributions",
        action="store_true",
        help="Tambien genera histogramas y boxplots de scores por archivo de predicciones.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir: Path = args.results_dir

    if not results_dir.exists():
        print(f"Directorio no encontrado: {results_dir}")
        return

    generated = make_metric_plots(results_dir)
    if args.score_distributions:
        generated.extend(make_score_distribution_plots(results_dir))

    print("\nListo.")
    if generated:
        print("PNGs generados:")
        for path in generated:
            print(f"- {path}")


if __name__ == "__main__":
    main()
