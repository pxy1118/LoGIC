"""Data-driven section 3.1 tables and figures, independent of manuscript editing."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from experiment import digest, read, save, verify, validate_result


COLORS = ["#356B8C", "#B47A32", "#53816A", "#886B96"]


def csv_write(path, rows):
    if not rows: return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def number(value):
    return "NA" if value is None else f"{value:.4g}"


def summarize(values):
    available = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return {"valid": len(available), "planned": len(values),
            "mean": float(available.mean()) if len(available) else None,
            "median": float(np.median(available)) if len(available) else None,
            "minimum": float(available.min()) if len(available) else None,
            "maximum": float(available.max()) if len(available) else None}


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    from matplotlib.font_manager import findfont
    font = findfont("Arial", fallback_to_default=True)
    mpl.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "mathtext.fontset": "stixsans", "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
        "legend.frameon": False, "text.usetex": False})
    return font


def export(fig, path):
    fig.savefig(str(path) + ".pdf")
    fig.savefig(str(path) + ".svg")
    fig.savefig(str(path) + ".png", dpi=600)


def letters(axes):
    for index, ax in enumerate(np.asarray(axes).ravel()):
        ax.annotate(chr(97 + index), (0, 1), xycoords="axes fraction", xytext=(-23, 9),
                    textcoords="offset points", weight="bold", fontsize=11, annotation_clip=False)


def geometry_plate(root, destination, lookup):
    import matplotlib.pyplot as plt
    import pyvista as pv
    cases = ["uniform_Gyroid_30", "uniform_Diamond_30", "uniform_Primitive_30",
             "topology_gradient", "density_gradient_z", "coupled_three_controls"]
    titles = ["Gyroid", "Diamond", "Primitive", "Topology gradient", "Density gradient", "Coupled controls"]
    rendered = destination / "geometry_views"; rendered.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.0))
    fig.subplots_adjust(left=.065, right=.985, bottom=.025, top=.91, wspace=.12, hspace=.30)
    for ax, name, title in zip(axes.ravel(), cases, titles):
        row = lookup[name]; path = root / "cases" / name / "surface.stl"
        if not path.exists():
            ax.text(.5, .5, "Generation rejected", ha="center", va="center", transform=ax.transAxes)
        else:
            plotter = pv.Plotter(off_screen=True, window_size=(800, 750))
            plotter.set_background("white")
            plotter.add_mesh(pv.read(path), color="#718B9A", smooth_shading=True,
                             ambient=.22, diffuse=.75, specular=.12)
            plotter.enable_parallel_projection(); plotter.view_isometric(); plotter.camera.zoom(1.05)
            picture = plotter.screenshot(str(rendered / f"{name}.png"), return_img=True)
            plotter.close(); ax.imshow(picture)
        ax.set_title(title, pad=10); ax.set_axis_off()
    letters(axes); export(fig, destination / "control_examples"); plt.close(fig)


def orientation_figure(root, destination, lookup):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.1))
    fig.subplots_adjust(left=.10, right=.97, bottom=.10, top=.94, wspace=.46, hspace=.51)
    for ax in axes.ravel(): ax.set_box_aspect(1)
    constant = [lookup[f"orientation_constant_{angle}"] for angle in (0, 15, 30, 45)]
    others = [lookup[name] for name in ("orientation_gradient_z_15", "orientation_gradient_z_30",
              "orientation_gradient_z_40", "orientation_gradient_z_60", "orientation_gradient_x_30",
              "orientation_gradient_y_30", "orientation_twist_z_30", "orientation_multiaxis")]
    means = [float(np.mean([r["orientation"]["local_z_error_mean_deg"] for r in constant]))] + [r["orientation"]["local_z_error_mean_deg"] for r in others]
    maxima = [max(r["orientation"]["local_z_error_max_deg"] for r in constant)] + [r["orientation"]["local_z_error_max_deg"] for r in others]
    y = np.arange(9)
    for i, (mean, maximum) in enumerate(zip(means, maxima)):
        axes[0, 0].plot([mean, maximum], [i, i], color=".75", lw=.7)
    axes[0, 0].scatter(means, y, color=COLORS[0], s=17, label="Mean", zorder=3)
    axes[0, 0].scatter(maxima, y, color=COLORS[1], s=17, label="Maximum", zorder=3)
    axes[0, 0].axvline(10, color=".65", lw=.7, ls="--")
    axes[0, 0].set(yticks=y, yticklabels=["Constant (4)", "z: 0–15°", "z: 0–30°", "z: 0–40°", "z: 0–60°*", "x: 0–30°", "y: 0–30°", "Twist z", "Multiaxis"],
                   xlabel="Axis error (deg)", title="Prescribed orientation fields", xlim=(-.4, max(maxima) * 1.12))
    axes[0, 0].invert_yaxis(); axes[0, 0].legend(loc="upper right")
    gradient = [lookup[f"orientation_gradient_z_{value}"] for value in (15, 30, 40, 60)]
    spans = [15, 30, 40, 60]
    for key, label, color in (("local_z_error_mean_deg", "Mean", COLORS[0]), ("local_z_error_max_deg", "Maximum", COLORS[1])):
        axes[0, 1].plot(spans, [r["orientation"].get(key, np.nan) for r in gradient], "o-", color=color, label=label, ms=4)
    axes[0, 1].set(xlabel="Orientation span (deg)", ylabel="Axis error (deg)", title="Graded orientation")
    axes[0, 1].legend(loc="upper left")
    for key, label, color in (("min_phase_stretch", "Minimum", COLORS[0]), ("max_phase_stretch", "Maximum", COLORS[1])):
        axes[1, 0].plot(spans, [r["orientation"].get(key, np.nan) for r in gradient], "o-", color=color, label=label, ms=4)
    axes[1, 0].axhline(1, color=".6", lw=.7, ls="--")
    axes[1, 0].set(xlabel="Orientation span (deg)", ylabel=r"Phase stretch $\sigma(J)$", title="Mapping deformation")
    axes[1, 0].legend(loc="lower left")
    name = "orientation_gradient_z_40"; ax = axes[1, 1]
    with np.load(root / "cases" / name / "geometry.npz") as values: mask = values["mask"]
    with np.load(root / "cases" / name / "orientation_nodes.npz") as data:
        indices = data["indices"]; mid = list(indices).index(mask.shape[1] // 2)
        positions = data["positions_mm"]; target = data["target_axes"]; actual = data["actual_axes"]
        sel = (slice(1, -1, 2), mid, slice(1, -1, 2))
        ax.imshow(mask[:, mask.shape[1] // 2, :].T, origin="lower", extent=(0, 7.5, 0, 7.5), cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
        for field, color, width in ((target, COLORS[0], .012), (actual, COLORS[1], .006)):
            ax.quiver(positions[sel][..., 0], positions[sel][..., 2], field[sel][..., 0, 2], field[sel][..., 2, 2],
                      color=color, angles="xy", scale_units="xy", scale=1.7, width=width, pivot="mid")
    ax.set(xlabel=r"$x$ (mm)", ylabel=r"$z$ (mm)", title="Target and realized axes")
    ax.set_aspect("equal")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=COLORS[0], lw=1.5), Line2D([], [], color=COLORS[1], lw=1.5)],
               ["Target axis", "Realized axis"], loc="lower center", bbox_to_anchor=(.75, 0), ncol=2, handlelength=1.2)
    letters(axes); export(fig, destination / "orientation_accuracy"); plt.close(fig)


def density_figure(root, destination, generated, lookup):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.1))
    fig.subplots_adjust(left=.10, right=.97, bottom=.10, top=.94, wspace=.46, hspace=.51)
    for ax in axes.ravel(): ax.set_box_aspect(1)
    topologies = ["Gyroid", "Diamond", "Primitive", "Gyroid+Diamond"]
    for color, topology in zip(COLORS, topologies):
        values = sorted([r for r in generated if r.get("section") == "uniform_density" and r["resolution"] == 240 and r["topology"] == topology], key=lambda r: r["density"]["target_volume_mean"])
        targets = [r["density"]["target_volume_mean"] for r in values]
        axes[0, 0].plot(targets, [r["surface"]["mesh_svf"] for r in values], "o-", color=color, label=topology.replace("Gyroid+Diamond", "G + D"), ms=3.5)
    axes[0, 0].plot([.23, .52], [.23, .52], color=".65", lw=.7, ls="--")
    axes[0, 0].set(xlabel=r"Target SVF $\rho$", ylabel="Surface-volume SVF", title="Uniform density")
    axes[0, 0].legend(loc="upper left")
    spatial_names = ["topology_gradient", "density_gradient_x", "density_gradient_y", "density_gradient_z", "coupled_three_controls", "coupled_three_topologies"]
    spatial_labels = ["Topology", "Density x", "Density y", "Density z", "All fields", "G/D/P + R"]
    for i, name in enumerate(spatial_names):
        row = lookup[name]["density"]
        axes[0, 1].plot([abs(row["global_error_pp"]), row["regional_mae_pp"]], [i, i], color=".75", lw=.8)
    axes[0, 1].scatter([abs(lookup[name]["density"]["global_error_pp"]) for name in spatial_names], np.arange(6), s=22, color=COLORS[0], label="Global")
    axes[0, 1].scatter([lookup[name]["density"]["regional_mae_pp"] for name in spatial_names], np.arange(6), s=22, color=COLORS[1], label="Regional")
    axes[0, 1].set(yticks=np.arange(6), yticklabels=spatial_labels, xlabel="SVF absolute error (pp)", title="Spatial control and coupling")
    axes[0, 1].invert_yaxis(); axes[0, 1].legend(loc="center right")
    all_pairs = []
    for name in spatial_names:
        color = COLORS[0] if name == "topology_gradient" else COLORS[2] if name.startswith("density_gradient") else COLORS[1]
        values = read(root / "cases" / name / "density.json")["regions"]
        truth = [r["target_svf"] for r in values]; prediction = [r["measured_svf"] for r in values]
        axes[1, 0].scatter(truth, prediction, s=12, color=color, alpha=.75)
        all_pairs += truth + prediction
    low, high = min(all_pairs) - .015, max(all_pairs) + .015
    axes[1, 0].plot([low, high], [low, high], color=".65", lw=.7, ls="--")
    axes[1, 0].set(xlabel="Regional target SVF", ylabel="Regional measured SVF", title="One-cell volume windows", xlim=(low, high), ylim=(low, high))
    from matplotlib.lines import Line2D
    axes[1, 0].legend([Line2D([], [], marker="o", linestyle="none", color=COLORS[k], markersize=4) for k in (0, 2, 1)],
                      ["Topology", "Density", "Coupled"], loc="upper left")
    with np.load(root / "cases/coupled_three_controls/profiles.npz") as data:
        axes[1, 1].plot(data["2_coordinate_mm"], data["2_target_svf"], color=".3", ls="--", lw=1, label="Target")
        axes[1, 1].plot(data["2_coordinate_mm"], data["2_measured_svf"], color=COLORS[1], lw=1, label="Measured")
    axes[1, 1].set(xlabel=r"$z$ (mm)", ylabel="Slab-averaged SVF", title="All-field coupled example")
    axes[1, 1].legend(loc="upper left")
    letters(axes); export(fig, destination / "density_accuracy"); plt.close(fig)


def quality_figure(destination, reused):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))
    fig.subplots_adjust(left=.10, right=.97, bottom=.20, top=.88, wspace=.46)
    solid6 = (1 - np.array([r["integrity"]["solid_6"]["largest_fraction"] for r in reused])) * 1e6
    solid26 = (1 - np.array([r["integrity"]["solid_26"]["largest_fraction"] for r in reused])) * 1e6
    ranks = np.arange(len(reused)) + 1
    axes[0].scatter(ranks, solid6, s=14, color=COLORS[0], label="6 neighbors")
    axes[0].scatter(ranks, solid26, s=22, facecolors="none", edgecolors=COLORS[1], linewidths=.7, label="26 neighbors")
    axes[0].set_ylim(-.6, max(1, solid6.max(), solid26.max()) * 1.12)
    axes[0].set(xlabel="Structure index", ylabel="Other solid components (ppm)", title="Solid-component residuals")
    axes[0].legend(loc="upper right")
    fractions = (1 - np.array([r["integrity"]["pore_6"]["spanning_fraction_xyz"] for r in reused])) * 100
    offsets = np.linspace(-.13, .13, len(reused))
    for row, offset in zip(fractions, offsets): axes[1].plot(np.arange(3) + offset, row, color=".75", lw=.45, alpha=.55)
    for axis, color in enumerate(COLORS[:3]): axes[1].scatter(axis + offsets, fractions[:, axis], color=color, s=12, zorder=3)
    axes[1].set(xticks=[0, 1, 2], xticklabels=[r"$x$", r"$y$", r"$z$"], xlabel="Spanning direction", ylabel="Nonspanning pore volume (%)", title="Pore-path residuals", ylim=(-.002, max(.01, fractions.max()) * 1.12))
    letters(axes); export(fig, destination / "geometry_integrity"); plt.close(fig)


def make_report(root):
    root = Path(root).resolve(); protocol = verify(root)
    if read(root / "state.json")["status"] not in ("MEASUREMENTS_COMPLETE", "COMPLETE"):
        raise RuntimeError("Wait until every planned geometry has a result or retained failure")
    identity = digest(root / "protocol.json")
    output = root / "report"; output.mkdir(exist_ok=True)
    generated, reused, failed = [], [], []
    lookup = {}
    for case in protocol["cases"]:
        folder = root / "cases" / case["case_id"]
        if (folder / "result.json").exists():
            result = validate_result(folder / "result.json", identity); generated.append(result)
        else:
            failure = read(folder / "failure.json"); failed.append({"case_id": case["case_id"], **failure})
            orientation = read(folder / "orientation.json") if (folder / "orientation.json").exists() else {}
            result = {**case, "status": "FAILED_RETAINED", "orientation": orientation.get("full_node", {}), "orientation_detail": orientation.get("sparse", {})}
        lookup[case["case_id"]] = result
    for case in protocol["reuse"]:
        path = root / "cases" / f"reuse_{case['sample_id']}" / "result.json"
        if path.exists(): reused.append(validate_result(path, identity))
        else: failed.append({"case_id": f"reuse_{case['sample_id']}", **read(path.parent / "failure.json")})
    orientation_rows, density_rows, quality_rows, regions = [], [], [], []
    for row in lookup.values():
        case_id = row["case_id"]
        if row.get("section") == "orientation":
            orientation_rows.append({"case_id": case_id, "status": row["status"], "resolution": row["resolution"],
                                     **row.get("orientation", {}), **row.get("orientation_detail", {})})
        if row["status"] != "COMPLETE": continue
        density_rows.append({"case_id": case_id, "section": row["section"], "topology": row["topology"],
                             "resolution": row["resolution"], **row["density"], "mesh_svf": row["surface"]["mesh_svf"]})
    for row in generated + reused:
        folder = root / "cases" / row["case_id"]
        for region in read(folder / "density.json")["regions"]:
            regions.append({"case_id": row["case_id"], "kind": row["kind"], **region})
        q = {"case_id": row["case_id"], "kind": row["kind"], "topology": row["topology"]}
        for phase in ("solid_6", "pore_6", "solid_26", "pore_26"):
            if phase not in row["integrity"]: continue
            value = row["integrity"][phase]
            q.update({f"{phase}_components": value["components"], f"{phase}_largest_fraction": value["largest_fraction"]})
            q.update({f"{phase}_spanning_{axis}": fraction for axis, fraction in zip("xyz", value["spanning_fraction_xyz"])})
        if row.get("surface"):
            q.update({key: value for key, value in row["surface"].items() if not isinstance(value, (list, dict))})
        quality_rows.append(q)
    csv_write(output / "orientation.csv", orientation_rows)
    csv_write(output / "density.csv", density_rows)
    csv_write(output / "coupling.csv", [row for row in density_rows if row["section"] == "coupled" and row["resolution"] == 240])
    csv_write(output / "regional_density.csv", regions)
    csv_write(output / "quality.csv", quality_rows)
    csv_write(output / "reused_density.csv", [{"sample_id": r["sample_id"], "case_id": r["case_id"],
              "topology": r["topology"], "sample_kind": r["sample_kind"], **r["density"]} for r in reused])
    group_rows = []
    for grouping in ("topology", "sample_kind"):
        for group in sorted({r[grouping] for r in reused}):
            members = [r for r in reused if r[grouping] == group]
            group_rows.append({"grouping": grouping, "group": group, "structures": len(members),
               "solid_largest_fraction_6_mean": float(np.mean([r["integrity"]["solid_6"]["largest_fraction"] for r in members])),
               "pore_spanning_z_fraction_6_mean": float(np.mean([r["integrity"]["pore_6"]["spanning_fraction_xyz"][2] for r in members])),
               "regional_density_mae_pp_mean": float(np.mean([r["density"]["regional_mae_pp"] for r in members]))})
    csv_write(output / "reused_groups.csv", group_rows)
    csv_write(output / "resources.csv", [{"case_id": r["case_id"], "kind": r["kind"], "elapsed_s": r["elapsed_s"], "peak_RAM_GiB": r["peak_RAM_GiB"]} for r in generated + reused])
    save(output / "failures.json", failed)
    resolution = []
    for case in protocol["cases"]:
        if "resolution_reference" not in case: continue
        low, high = lookup[case["case_id"]], lookup[case["resolution_reference"]]
        if low["status"] != "COMPLETE" or high["status"] != "COMPLETE": continue
        row = {"case_id": case["resolution_reference"]}
        for name, value in (("n120", low), ("n240", high)):
            row.update({f"{name}_mesh_svf": value["surface"]["mesh_svf"], f"{name}_regional_mae_pp": value["density"]["regional_mae_pp"],
                        f"{name}_mean_axis_error_deg": value["orientation"].get("local_z_error_mean_deg")})
        resolution.append(row)
    csv_write(output / "resolution.csv", resolution)
    uniform = [r for r in generated if r["section"] == "uniform_density" and r["resolution"] == 240]
    gradient = [lookup[f"density_gradient_{axis}"] for axis in "xyz"]
    constant = [lookup[f"orientation_constant_{value}"] for value in (0, 15, 30, 45)]
    main = [r for r in generated if r["resolution"] == 240]
    summary = {"protocol_sha256": identity, "generated_completed": len(generated), "generated_planned": len(protocol["cases"]),
        "reused_completed": len(reused), "reused_planned": len(protocol["reuse"]), "failures": len(failed),
        "uniform_surface_absolute_error_pp": summarize([abs(r["density"]["mesh_error_pp"]) if r["density"]["mesh_error_pp"] is not None else None for r in uniform]),
        "uniform_sampled_absolute_error_pp": summarize([abs(r["density"]["global_error_pp"]) for r in uniform]),
        "gradient_regional_mae_pp": summarize([r.get("density", {}).get("regional_mae_pp") for r in gradient]),
        "gradient_half_cell_regional_mae_pp": summarize([r.get("density", {}).get("regional_mae_pp_half_cell") for r in gradient]),
        "uniform_mix_half_cell_regional_mae_pp": lookup["uniform_Gyroid_Diamond_35"]["density"]["regional_mae_pp_half_cell"],
        "topology_gradient_regional_mae_pp": lookup["topology_gradient"]["density"]["regional_mae_pp"],
        "coupled_three_controls_regional_mae_pp": lookup["coupled_three_controls"].get("density", {}).get("regional_mae_pp"),
        "constant_affine_error": summarize([r.get("orientation_detail", {}).get("constant_affine_max_error_generator_units") for r in constant]),
        "orientation_main_max_axis_error_deg": max(r["orientation"].get("local_z_error_max_deg", 0) for r in lookup.values() if r["section"] == "orientation" and r["resolution"] == 240),
        "orientation_main_above_10_deg": [r["case_id"] for r in lookup.values() if r["section"] == "orientation" and r["resolution"] == 240 and r["orientation"].get("local_z_error_max_deg", 0) > 10],
        "new_closed_surfaces": sum(r["surface"]["boundary_edges"] == 0 and r["surface"]["nonmanifold_edges"] == 0 for r in main),
        "new_main_completed": len(main),
        "new_main_multiple_solid_components_6": sum(r["integrity"]["solid_6"]["components"] > 1 for r in main),
        "new_main_largest_solid_fraction_6": summarize([r["integrity"]["solid_6"]["largest_fraction"] for r in main]),
        "new_main_AR_below_5_fraction": summarize([r["surface"]["aspect_ratio_below_5_fraction"] for r in main]),
        "reused_solid_largest_fraction_6": summarize([r["integrity"]["solid_6"]["largest_fraction"] for r in reused]),
        "reused_pore_spanning_z_6": summarize([r["integrity"]["pore_6"]["spanning_fraction_xyz"][2] for r in reused]),
        "reused_density_regional_mae_pp": summarize([r["density"]["regional_mae_pp"] for r in reused]),
        "experiment_unit": "structure; deterministic prescribed design; no node/window-level inferential CI",
        "source_scope": protocol["scope"]}
    save(output / "results.json", summary)
    save(output / "measurement_notes.json", {
        "new_mask_grid": "240 endpoint nodes across each physical 7.5 mm axis; sampled SVF is the unweighted occupancy fraction, surface SVF is measured separately.",
        "reused_grid": "240 initial-FE voxel cells, using each published origin/spacing; no source geometry is changed.",
        "regions": "Primary bins are fixed physical thirds (27 cell-scale regions); secondary bins are sixths (216 half-cell-scale regions). Mean target uses the same actual sample coordinates as occupancy.",
        "profile_window": "80 samples at n240; nominal sample-support width n*h is stored as window_mm; sample-center span is (n-1)*h. This is approximately one 2.5 mm cell, not a sub-cell density guarantee.",
        "quality_display": "Figures show exact complements of stored connectivity fractions: other-solid volume in ppm, nonspanning pore volume in percent. No structure is removed.",
        "surface_stage": "Raw initial isosurface triangulation, not a 3-matic remesh or tetrahedral mesh. AR statistics cannot be compared directly across mesh stages."})
    font = style()
    figures = output / "figures"; figures.mkdir(exist_ok=True)
    save(figures / "figure_contract.json", {"font": font, "backend": "python", "dpi": 600,
        "control_examples": "Image plate: actual unmodified initial surface geometry; shows separate/coupled editable fields.",
        "orientation_accuracy": "All 12 prescribed orientation cases (four constants grouped), graded-span trend, phase deformation and an actual mid-plane realization.",
        "density_accuracy": "Quantitative grid: exported global volume, paired global/regional error for all six spatial cases, all regional observations and the preselected fully coupled profile.",
        "geometry_integrity": "Paired per-structure residual fractions: voxels outside the largest solid component and pore volume not spanning opposite faces among 50 retained structures.",
        "sampling": "All planned cases in tables, failed cases retained. Geometry examples and sparse arrow positions fixed before viewing errors.",
        "uncertainty": "No CI/error bars for deterministic case grids or repeated nodes; all per-structure observations retained.",
        "allocation_note": "Density panels include all six planned spatial cases and orientation panels include all 12 fields so conclusion-changing coupling/axis errors are visible. Protocol, measurements and primary windows were unchanged.",
        "manual_visual_review": "Required after final scientific rendering; automated audit does not replace it."})
    geometry_plate(root, figures, lookup)
    orientation_figure(root, figures, lookup)
    density_figure(root, figures, generated, lookup)
    quality_figure(figures, reused)
    lines = ["# Section 3.1 geometry experiment results", "",
        f"Completed generated cases: {len(generated)}/{len(protocol['cases'])}; reused initial geometries: {len(reused)}/{len(protocol['reuse'])}; failures retained: {len(failed)}.", "",
        "## Controlled orientation",
        f"Constant-field affine coordinate error (maximum across cases): {number(summary['constant_affine_error']['maximum'])} generator units.",
        "| Graded y-rotation along z | Mean axis error / deg | Maximum / deg | Phase stretch range | Minimum det(J) |",
        "|---|---:|---:|---|---:|"]
    for angle in (15, 30, 40, 60):
        q = lookup[f"orientation_gradient_z_{angle}"]["orientation"]
        lines.append(f"| 0–{angle}° | {number(q.get('local_z_error_mean_deg'))} | {number(q.get('local_z_error_max_deg'))} | {number(q.get('min_phase_stretch'))}–{number(q.get('max_phase_stretch'))} | {number(q.get('min_determinant'))} |")
    lines += ["", "The 60° span exceeds the dataset's 20° neighboring-control bound and is a declared stress case. Axes are phase-coordinate directions, not surface normals or FEA stress directions.",
        f"The largest axis error across all 12 prescribed fields is {summary['orientation_main_max_axis_error_deg']:.4g}°. Fields exceeding the dataset's 10° output diagnostic threshold: {', '.join(summary['orientation_main_above_10_deg'])}. A bounded neighboring-angle change alone does not guarantee an accurate compatible realization.",
        "", "## Density accuracy",
        f"Uniform-density exported-surface SVF absolute error: mean {number(summary['uniform_surface_absolute_error_pp']['mean'])} pp; maximum {number(summary['uniform_surface_absolute_error_pp']['maximum'])} pp; valid {summary['uniform_surface_absolute_error_pp']['valid']}/20.",
        f"Uniform sampled-node error: mean {number(summary['uniform_sampled_absolute_error_pp']['mean'])} pp. Quantile thresholding makes uniform sampled fraction accurate by construction; exported surface and regional errors are separate checks.",
        f"Across the three prescribed axial density gradients, mean per-structure regional MAE: {number(summary['gradient_regional_mae_pp']['mean'])} pp. Regions are 3×3×3 one-cell windows; half-cell sensitivity is retained, not selected after results.",
        f"The topology-gradient case has regional MAE {number(summary['topology_gradient_regional_mae_pp'])} pp despite an accurately controlled global SVF. The all-field coupled case has regional MAE {number(summary['coupled_three_controls_regional_mae_pp'])} pp. These are conclusion-changing coupling effects, included in the main density figure and coupling.csv rather than excluded from the accuracy claim.",
        f"Half-cell regional MAE for axial gradients averages {number(summary['gradient_half_cell_regional_mae_pp']['mean'])} pp; the matched uniform 50:50 G+D case at SVF 0.35 already has {number(summary['uniform_mix_half_cell_regional_mae_pp'])} pp at this sub-cell scale. Intrinsic within-cell material oscillation must be distinguished from errors in the cell-averaged control target.",
        "", "## Geometric integrity",
        f"Closed exported initial surfaces: {summary['new_closed_surfaces']}/{summary['new_main_completed']} completed main cases. Closedness does not certify self-intersection freedom, manufacturability or volume-mesh quality.",
        f"Raw masks have multiple 6-connected solid components in {summary['new_main_multiple_solid_components_6']}/{summary['new_main_completed']} cases; minimum largest-solid fraction is {number(summary['new_main_largest_solid_fraction_6']['minimum'])}. Raw initial triangle AR<5 fraction averages {number(summary['new_main_AR_below_5_fraction']['mean'])}; this is not a remeshed-surface statistic and does not reproduce the old manuscript's 99.95% mesh claim.",
        f"Reused-geometry largest solid-component fraction (6-neighbor) minimum: {number(summary['reused_solid_largest_fraction_6']['minimum'])}; z-spanning pore-volume fraction minimum: {number(summary['reused_pore_spanning_z_6']['minimum'])}.",
        "The 50 reused geometries were selected by topology/sample kind before measurement from the completed 433-member cohort. This is conditional geometry characterization, not a whole-campaign generation success rate. Original surface meshes were unavailable in retained source-run paths; no voxel-reconstructed surface was substituted for their triangle statistics.",
        "", "## Interpretation and scope",
        "Inputs can be prescribed independently, but geometric outputs are not proven mathematically orthogonal. Small global SVF error does not certify local density accuracy or distortion-free orientation. Report actual coupled-case changes in density, orientation and connectivity.",
        "The three 120³/240³ pairs are a limited geometric resolution check, not FEA mesh convergence. No solver, model training, surrogate test evaluation or manuscript editing was performed.",
        "", "## Evidence files", "",
        "orientation.csv, density.csv, regional_density.csv, quality.csv, resolution.csv, resources.csv, failures.json and results.json are the numerical sources. Each generated case retains controls, sampled solid mask, STL, regional data, sparse axes and file hashes. Four figures provide PDF/SVG/600-dpi PNG exports."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "CAPTIONS_EN.md").write_text(
        "# Geometry-validation figure captions\n\n"
        "**Control examples.** Actual initial surfaces generated using the frozen LoGIC implementation, shown at the same nominal specimen size. Top row: Gyroid, Diamond and Primitive at uniform target SVF 0.30. Bottom row: topology grading, density grading, and coupled topology–density–orientation control. No surface smoothing or component removal was applied. Display lighting changes appearance only.\n\n"
        "**Orientation accuracy.** (a) Full-node mean and maximum local-z axis error across all 12 prescribed fields; the four constant rotations have identical zero angular error and are grouped in one row. The dashed line is the dataset's 10° output-screening threshold, not an exclusion used in this experiment. (b) Error versus orientation span for the four y-rotation fields graded along z. (c) Minimum and maximum phase-Jacobian singular values; physical coordinate stretch is related to their reciprocals. (d) Actual solid mid-plane for the 0–40° case, with target axes in blue and realized axes in ochre at fixed sparse positions. The starred 60° case is a prespecified stress test beyond the dataset's neighboring-angle range; other fields can also exceed the output-error threshold. Points represent deterministic configurations, not independent spatial replicates.\n\n"
        "**Density accuracy.** (a) Exported-surface SVF for four topology settings and five uniform density levels (20 structures). (b) Paired global sampled-SVF absolute error and regional MAE for all six prescribed spatial/coupled cases. Errors are percentage points. (c) All regional measured/target pairs from these six structures, using 27 fixed cell-scale regions per structure and the exact matching mean trilinear target. (d) Cell-scale sliding-slab target/measured profiles for the prespecified all-field coupled example, using the complete cross-section and excluding edge padding. Regional windows are repeated measurements, not independent structures.\n\n"
        "**Geometric integrity.** Measurements on 50 prespecified, stratified initial voxel geometries from the completed FEA cohort. (a) Solid-volume fraction outside the largest component, in parts per million, under 6- and 26-neighbor connectivity. (b) Pore-volume fraction not belonging to any component that intersects both opposite faces in each direction, using 6-neighbor connectivity. These residuals are exact complements of the connectivity fractions retained in the source tables. Lines link directions within the same structure; horizontal offsets separate overlapping points. Multiple independent spanning pore components are allowed. These measurements do not estimate unfiltered generation success or certify final volume meshes.\n", encoding="utf-8")
    save(root / "report_manifest.json", {"protocol_sha256": identity, "plotting_source_sha256": digest(Path(__file__)),
        "artifacts": {p.relative_to(root).as_posix(): digest(p) for p in output.rglob("*") if p.is_file()}})
    state = read(root / "state.json"); state.update(status="COMPLETE", phase="COMPLETE")
    save(root / "state.json", state)
    print(json.dumps(summary, indent=2))
