"""
Genera gráficos de calidad académica para los scores de similitud de retrieval.

Lee todos los archivos JSON y CSV que contengan la clave/columna `similarity_score`
dentro de un directorio de resultados y produce, por cada archivo:
  - KDE + histograma de la distribución de scores
  - Boxplot de los scores

Los gráficos se guardan como PNG (300 dpi) en la misma carpeta que los resultados.

Uso:
    python -m src.plot_retrieval_scores
    python -m src.plot_retrieval_scores --results-dir outputs/results/dataset_enriched/retrieval_textual
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── estilo académico global ────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)

SCORE_KEY = "similarity_score"
DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "results"
    / "dataset_enriched"
    / "retrieval_textual"
)


# ── I/O ────────────────────────────────────────────────────────────────────────


def load_scores(path: Path) -> np.ndarray:
    """Extrae los valores de `similarity_score` de un JSON o CSV."""
    if path.suffix == ".json":
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            df = pd.DataFrame(raw)
        elif isinstance(raw, dict):
            # Soporta {results: [...]} u otras envolturas con una lista anidada.
            lists = [v for v in raw.values() if isinstance(v, list)]
            if not lists:
                raise ValueError(f"No se encontró una lista en {path}")
            df = pd.DataFrame(lists[0])
        else:
            raise ValueError(f"Formato JSON no reconocido en {path}")
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Extensión no soportada: {path.suffix}")

    if SCORE_KEY not in df.columns:
        raise KeyError(f"Columna '{SCORE_KEY}' no encontrada en {path}")

    return df[SCORE_KEY].dropna().to_numpy(dtype=float)


def collect_result_files(results_dir: Path) -> list[Path]:
    files = sorted(
        [p for p in results_dir.glob("*.json") if not p.stem.startswith(".")]
        + [p for p in results_dir.glob("*.csv") if not p.stem.startswith(".")]
    )
    return files


# ── gráficos ───────────────────────────────────────────────────────────────────


def plot_kde_hist(
    scores: np.ndarray,
    label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))

    # Histograma con normalización a densidad
    ax.hist(
        scores,
        bins=30,
        density=True,
        color="#4C72B0",
        alpha=0.35,
        edgecolor="white",
        linewidth=0.6,
        label="Histograma",
    )

    # KDE encima
    sns.kdeplot(scores, ax=ax, color="#4C72B0", linewidth=2, label="KDE")

    # Líneas de estadísticos descriptivos
    mean_val = float(np.mean(scores))
    median_val = float(np.median(scores))
    ax.axvline(mean_val, color="#DD4444", linestyle="--", linewidth=1.4,
               label=f"Media = {mean_val:.3f}")
    ax.axvline(median_val, color="#228B22", linestyle=":", linewidth=1.4,
               label=f"Mediana = {median_val:.3f}")

    ax.set_xlabel("Cosine Similarity Score", labelpad=8)
    ax.set_ylabel("Densidad", labelpad=8)
    ax.set_title(f"Distribución de Scores — {label}", pad=12)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {output_path}")


def plot_boxplot(
    scores: np.ndarray,
    label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 5))

    bp = ax.boxplot(
        scores,
        vert=True,
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="#DD4444", linewidth=2),
        boxprops=dict(facecolor="#4C72B0", alpha=0.55),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4, markerfacecolor="#4C72B0"),
    )

    # Anotaciones de estadísticos
    q1, med, q3 = np.percentile(scores, [25, 50, 75])
    ax.text(
        1.28, med, f"Mediana\n{med:.3f}",
        va="center", ha="left", fontsize=8, color="#DD4444",
        transform=ax.get_yaxis_transform(),
    )

    ax.set_xticks([])
    ax.set_ylabel("Cosine Similarity Score", labelpad=8)
    ax.set_title(f"Boxplot de Scores — {label}", pad=12)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {output_path}")


def print_stats(scores: np.ndarray, label: str) -> None:
    q1, q3 = np.percentile(scores, [25, 75])
    print(
        f"\n[{label}] n={len(scores)}"
        f"  media={np.mean(scores):.4f}"
        f"  mediana={np.median(scores):.4f}"
        f"  std={np.std(scores):.4f}"
        f"  IQR=[{q1:.4f}, {q3:.4f}]"
        f"  min={np.min(scores):.4f}"
        f"  max={np.max(scores):.4f}"
    )


# ── main ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera gráficos de scores de retrieval")
    p.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Carpeta con archivos JSON/CSV de resultados (default: dataset_enriched/retrieval_textual)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir: Path = args.results_dir

    if not results_dir.exists():
        print(f"Directorio no encontrado: {results_dir}")
        print("Crea los resultados de retrieval primero o pasa --results-dir <ruta>.")
        return

    files = collect_result_files(results_dir)
    if not files:
        print(f"No se encontraron archivos JSON/CSV en {results_dir}")
        return

    print(f"Procesando {len(files)} archivo(s) en {results_dir}\n")

    for path in files:
        try:
            scores = load_scores(path)
        except (KeyError, ValueError) as exc:
            print(f"  Saltando {path.name}: {exc}")
            continue

        label = path.stem
        print_stats(scores, label)

        plot_kde_hist(
            scores,
            label=label,
            output_path=results_dir / f"{label}_kde_hist.png",
        )
        plot_boxplot(
            scores,
            label=label,
            output_path=results_dir / f"{label}_boxplot.png",
        )

    print("\nListo.")


if __name__ == "__main__":
    main()
