"""Selected-model tables, structural bootstrap intervals and Python figures."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from data import digest, read, save
from experiment import write_csv


ERROR_KEYS = ["relative_velocity_l2", "relative_pressure_l2", "velocity_rmse_m_s", "pressure_rmse_Pa",
              "lnK_mae", "lnDP_mae", "K_mape_percent", "DP_mape_percent"]


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return [{key: int(value) if key in ("sample_id", "valid_grid_points") else float(value)
                 for key, value in row.items()} for row in csv.DictReader(stream)]


def structural_intervals(rows, repetitions, seed):
    """Resample entire structures, keeping all outputs for one structure paired."""
    values = np.array([[row[key] for key in ERROR_KEYS] for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(rows), size=(repetitions, len(rows)))
    boot = values[index].mean(1)
    limits = np.quantile(boot, [.025, .975], axis=0)
    return {key: {"mean": float(values[:, i].mean()), "CI95": limits[:, i].tolist(),
                  "unit": "structure", "n": len(rows)} for i, key in enumerate(ERROR_KEYS)}


def plot_style(config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    from matplotlib import font_manager
    requested = config["report"].get("font_family", "Arial")
    font_path = font_manager.findfont(requested, fallback_to_default=True)
    mpl.rcParams.update({"font.family": requested, "font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "mathtext.fontset": "stixsans",
                         "text.usetex": False, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .7, "legend.frameon": False})
    return font_path


def export_figure(fig, base, config, colorbars=()):
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".svg")
    fig.savefig(str(base) + ".png", dpi=config["report"]["dpi"])


def make_figures(root, rows, config, protocol):
    import matplotlib.pyplot as plt
    font_path = plot_style(config)
    output = root / "report/figures"; output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25))
    fig.subplots_adjust(left=.10, right=.98, bottom=.21, top=.88, wspace=.40)
    for axis, prefix, multiplier, symbol, unit, letter in zip(
            axes, ("K", "DP"), (1e9, 1), (r"$K$", r"$\Delta p$"),
            (r"$10^{-9}$ m$^2$", "Pa"), ("a", "b")):
        truth = np.array([row[f"{prefix}_true"] for row in rows]) * multiplier
        predicted = np.array([row[f"{prefix}_pred"] for row in rows]) * multiplier
        high = float(max(truth.max(), predicted.max()) * 1.08)
        axis.plot([0, high], [0, high], color=".6", linewidth=.8, zorder=1)
        axis.scatter(truth, predicted, s=13, color="#36789c", alpha=.75, edgecolors="none", zorder=2)
        axis.set(xlim=(0, high), ylim=(0, high), xlabel=f"CFD {symbol} ({unit})", ylabel=f"Predicted {symbol} ({unit})")
        axis.set_aspect("equal", adjustable="box")
        axis.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-24, 9), textcoords="offset points", weight="bold", va="bottom")
    export_figure(fig, output / "scalar_parity", config); plt.close(fig)
    snapshot = root / "snapshot"
    for sid in protocol["representative_test_ids"]:
        with np.load(snapshot / "samples" / str(sid) / "field.npz") as reference:
            truth = np.stack([reference[key] for key in ("u", "v", "w", "p")])
            mask = reference["mask"]
            origin, spacing = reference["origin_mm"], reference["spacing_mm"]
        truth[3] -= read(snapshot / "samples" / str(sid) / "metrics.json")["pressure_outlet_Pa"]
        with np.load(root / "test/ensemble/predictions" / f"{sid}.npz") as predicted:
            prediction = predicted["field"]
        z = truth.shape[-1] // 2
        true_speed = np.linalg.norm(truth[:3, :, :, z], axis=0) * 1000
        pred_speed = np.linalg.norm(prediction[:3, :, :, z], axis=0) * 1000
        planes = [(true_speed, pred_speed, "viridis", r"$|\mathbf{u}|$ (mm s$^{-1}$)"),
                  (truth[3, :, :, z], prediction[3, :, :, z], "cividis", r"$p$ (Pa)")]
        fig = plt.figure(figsize=(7.1, 4.9))
        grid = fig.add_gridspec(2, 6, width_ratios=[1, .05, 1, .05, 1, .05],
                               left=.085, right=.92, bottom=.105, top=.9, wspace=.48, hspace=.50)
        bars = []
        extent = [float(origin[0]), float(origin[0] + spacing[0] * 63),
                  float(origin[1]), float(origin[1] + spacing[1] * 63)]
        for row, (actual, estimated, cmap, units) in enumerate(planes):
            valid = mask[:, :, z]
            lo = float(min(np.nanmin(actual[valid]), np.nanmin(estimated[valid])))
            hi = float(max(np.nanmax(actual[valid]), np.nanmax(estimated[valid])))
            hi = max(hi, lo + 1e-8)
            error = np.abs(estimated - actual)
            for col, (values, color, minimum, maximum) in enumerate(
                    [(actual, cmap, lo, hi), (estimated, cmap, lo, hi),
                     (error, "magma", 0, max(float(error[valid].max()), 1e-8))]):
                axis = fig.add_subplot(grid[row, 2 * col])
                bar = fig.add_subplot(grid[row, 2 * col + 1]); bars.append(bar)
                image = axis.imshow(np.ma.array(values, mask=~valid).T, origin="lower", extent=extent,
                                    interpolation="nearest", cmap=color, vmin=minimum, vmax=maximum)
                fig.colorbar(image, cax=bar)
                bar.set_title(r"mm s$^{-1}$" if row == 0 else "Pa", fontsize=8.5, pad=7)
                axis.set(xlabel=r"$x$ (mm)", ylabel=r"$y$ (mm)" if col == 0 else "")
                axis.set_xticks([0, 2, 4, 6]); axis.set_yticks([0, 2, 4, 6])
                if col:
                    axis.set_yticklabels([])
                if row == 0:
                    axis.set_title(["CFD", "Prediction", "Absolute error"][col], fontsize=9, pad=12)
                axis.annotate(chr(97 + row * 3 + col), xy=(0, 1), xycoords="axes fraction", xytext=(0, 6),
                              textcoords="offset points", weight="bold", va="bottom")
        fig.canvas.draw()
        for image_axis, bar in zip([axis for axis in fig.axes if axis not in bars], bars):
            image_box, bar_box = image_axis.get_position(), bar.get_position()
            bar.set_position([bar_box.x0, image_box.y0, bar_box.width, image_box.height])
        export_figure(fig, output / f"field_{sid}", config, bars); plt.close(fig)
    save(output / "figure_contract.json", {"claim": "The selected surrogate reproduces engineering scalars and spatial flow patterns on held-out structures.",
                                             "archetypes": ["quantitative grid", "image plate + quant"],
                                             "source": "Every test structure in scalar parity; preregistered initial-metadata IDs for field examples.",
                                             "uncertainty": "Parity shows ensemble predictions per structure; seed SD and structural bootstrap CI are in the accompanying tables.",
                                             "field_processing": "z-mid plane; speed in mm/s, gauge pressure Pa; common reference/prediction color ranges; absolute-error scale separate.",
                                             "font_file": font_path, "backend": "python", "dpi": config["report"]["dpi"],
                                             "representative_ids": protocol["representative_test_ids"],
                                             "manual_visual_review": "Required for actual scientific figures before manuscript placement; pipeline fixture checks are not a scientific figure approval."})


def make_report(root):
    root = Path(root)
    from formal import completed_training, validate_evaluation
    config = read(root / "config.json"); completed_training(root, config)
    protocol = read(root / "test_protocol_frozen.json"); protocol_hash = digest(root / "test_protocol_frozen.json")
    output = root / "report"; output.mkdir(parents=True, exist_ok=True)
    seed_metrics = []
    for seed in config["formal"]["seeds"]:
        record = validate_evaluation(root / "test" / f"seed{seed}", protocol_hash)
        seed_metrics.append({"seed": seed, **record["metrics"]})
    ensemble = validate_evaluation(root / "test/ensemble", protocol_hash)
    rows = csv_rows(root / "test/ensemble/per_structure.csv")
    intervals = structural_intervals(rows, config["report"]["bootstrap_repetitions"], config["report"]["bootstrap_seed"])
    write_csv(output / "seed_metrics.csv", seed_metrics)
    statistics = {}
    for key in seed_metrics[0]:
        if key == "seed":
            continue
        values = [row[key] for row in seed_metrics]
        if all(value is not None for value in values):
            statistics[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
    save(output / "ensemble_intervals.json", intervals)
    group_rows = []
    snapshot = root / "snapshot"
    for group_key in ("topology", "kind"):
        groups = {}
        for row in rows:
            info = read(snapshot / "samples" / str(row["sample_id"]) / "sample.json")
            key = "+".join(info["topology_family"]) if group_key == "topology" else info["kind"]
            groups.setdefault(key, []).append(row)
        for name, members in groups.items():
            group_rows.append({"group_type": group_key, "group": name, "n": len(members),
                               **{key: float(np.mean([r[key] for r in members])) for key in ERROR_KEYS}})
    write_csv(output / "group_metrics.csv", group_rows)
    table = ["| Metric | Three-seed mean ± SD | Ensemble | Ensemble structural CI95 |", "|---|---:|---:|---:|"]
    for key in ERROR_KEYS:
        value = statistics[key]; lower, upper = intervals[key]["CI95"]
        table.append(f"| {key} | {value['mean']:.6g} ± {value['std']:.4g} | {ensemble['metrics'][key]:.6g} | [{lower:.6g}, {upper:.6g}] |")
    counts = read(snapshot / "snapshot.json")["counts"]
    fixture = bool(config.get("allow_synthetic_fixture", False))
    (output / "results.md").write_text(
        "# " + ("SYNTHETIC PIPELINE CHECK — NOT SCIENTIFIC RESULTS" if fixture else "Selected CFD surrogate results") + "\n\n"
        + f"Architecture: LoGIC-conditioned residual 3D U-Net. Splits: {counts}.\n\n"
        + "Scalar labels are the stored metrics. Fields use the saved 64³ target grid. All normalization was fitted on training members only.\n\n"
        + "\n".join(table) + "\n\nSeed variation and bootstrap uncertainty are separate. The bootstrap unit is a complete structure, not a field voxel."
        + " Figures and source tables concern only the selected method. Original source quality flags remain unchanged.\n", encoding="utf-8")
    if config["report"]["figures"]:
        make_figures(root, rows, config, protocol)
    summary = {"status": "COMPLETE", "synthetic_fixture": fixture, "protocol_sha256": protocol_hash,
               "counts": counts, "selected_variant": "none", "seed_statistics": statistics,
               "seed_metrics": seed_metrics, "ensemble": ensemble["metrics"], "ensemble_intervals": intervals,
               "bootstrap_unit": "structure", "no_method_comparison": True}
    save(root / "formal_summary.json", summary)
    save(root / "completion_manifest.json", {"protocol_sha256": protocol_hash,
                                             "artifacts": {str(path.relative_to(root).as_posix()): digest(path)
                                                           for path in [root / "formal_summary.json", *output.rglob("*")]
                                                           if path.is_file()}})
    return summary
