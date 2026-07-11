from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

DATASETS = {
    "Objetivo enriquecido": ROOT
    / "outputs/error_analysis/dataset_enriched/pairwise_effect_summary.csv",
    "Respuesta larga": ROOT
    / "outputs/error_analysis/dataset_longest_answer/pairwise_effect_summary.csv",
}

LABELS = {
    "lora_vs_zero_shot": "LoRA frente a base",
    "rag_on_zero_shot": "RAG sobre base",
    "rag_on_lora": "RAG sobre LoRA",
}


def load_overall_effects() -> pd.DataFrame:
    frames = []
    for dataset_name, path in DATASETS.items():
        frame = pd.read_csv(path)
        frame = frame[(frame["dimension"] == "overall") & (frame["stratum"] == "all")].copy()
        frame["dataset"] = dataset_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    effects = load_overall_effects()
    effect_order = list(LABELS)
    colors = {
        "Objetivo enriquecido": "#2F5D8C",
        "Respuesta larga": "#D17A3A",
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), constrained_layout=True)

    y_positions = range(len(effect_order))
    offsets = {"Objetivo enriquecido": -0.16, "Respuesta larga": 0.16}
    height = 0.28

    for dataset_name in DATASETS:
        subset = effects.set_index("effect").loc[effect_order]
        subset = subset[subset["dataset"] == dataset_name]
        positions = [position + offsets[dataset_name] for position in y_positions]

        axes[0].barh(
            positions,
            subset["mean_delta"],
            height=height,
            color=colors[dataset_name],
            label=dataset_name,
        )
        axes[1].barh(
            positions,
            subset["positive_rate"] * 100,
            height=height,
            color=colors[dataset_name],
        )

    labels = [LABELS[effect] for effect in effect_order]
    for axis in axes:
        axis.set_yticks(list(y_positions), labels)
        axis.invert_yaxis()
        axis.grid(axis="x", color="#D8D8D8", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)

    axes[0].axvline(0, color="#333333", linewidth=0.8)
    axes[0].set_xlim(-0.065, 0.11)
    axes[0].set_xlabel("Cambio medio del score compuesto")

    axes[1].axvline(50, color="#555555", linewidth=0.8, linestyle="--")
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Filas en las que mejora (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=2,
        frameon=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / "paired_effects.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / "paired_effects.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
