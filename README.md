# LoGIC

LoGIC represents heterogeneous Gyroid, Diamond and Primitive structures with spatially varying topology weights, density and orientation on a 3 × 3 × 3 control lattice. This directory contains the integrated-orientation generator, geometry-control experiments, simulation scripts, trained-model inference and paper figure code. The companion dataset repository's `main` branch contains the input geometries, simulation responses, fixed splits, ensemble weights and reference predictions. [LoGIC-Dataset](https://huggingface.co/datasets/pxy1118/LoGIC-Dataset)

| Application | Input | Prediction | Training / validation / test |
|---|---|---|---|
| Uniaxial compression | 351 control variables, 38 geometry descriptors, 80³ fractional occupancy from 240³ solid voxels | Eight POD coefficients reconstruct 1,001 stress values over 0–5% strain | 304 / 67 / 62 |
| Steady flow along z | Initial 240³ solid geometry resampled on the 64³ query grid and the control field | Three velocity components, outlet-referenced pressure, apparent permeability and pressure drop | 353 / 73 / 75 |

## Run a prediction

The release was tested on Python 3.11.14 with the packages in `requirements.txt`. The reference ensemble inference supports CPU and CUDA. For the following commands, start in this directory and place the downloaded `LoGIC-Dataset` repository alongside the `LoGIC` repository. For a different local layout, substitute the path to the dataset repository root.

```sh
python -m pip install -r requirements.txt
python ../LoGIC-Dataset/unpack.py --output data
python predict.py fea --sample data/fea/samples/61 --models ../LoGIC-Dataset/models --output outputs/fea61
python predict.py cfd --sample data/cfd/samples/38 --models ../LoGIC-Dataset/models --output outputs/cfd38
```

The FEA example writes `curve.csv` with dimensionless engineering strain, stress in MPa and the across-seed sample standard deviation, plus `metrics.json`. For a single member the standard deviation column is empty. The CFD example writes `field.npz` with velocity in m/s and gauge pressure in Pa, plus scalar `metrics.json`. Both examples use every ensemble member. Select one with `--seed 17` or choose GPU inference with `--device cuda`.

For a new FEA design, supply `controls.npz`, `features.json` and `voxel.npz` with the same initial geometry schema and physical protocol as the released cohort. For a new CFD design, supply `controls.npz`, `voxel.npz` and `--grid grid.json` with `origin_mm`, `spacing_mm` and `shape` in x, y, z order. Released CFD examples use their own coordinates in `query_grids.json`. The predictors read the initial inputs and grid geometry only.

## Reproduce the reported results

```sh
python check_results.py --release ../LoGIC-Dataset --data data
python figures.py --release ../LoGIC-Dataset --data data --output outputs/figures
```

The first command recomputes the primary ensemble metrics on all 62 mechanical and 75 flow test structures from saved predictions and simulation targets. The second redraws Figures 7–11 using the published data and sample IDs. To redraw only Figure 9, append `9`. The representative compression structures have public FEA IDs 61, 51 and 136, in ascending order of the three selected error ranks.

## Train or repeat the geometry experiment

Run these from this directory after unpacking both datasets into `data`. The training commands create fresh study directories. The formal training configurations use CUDA. The same study name resumes an interrupted run according to its frozen configuration.

```sh
python research/fea_prediction/main.py study run --study-id fea_reproduction --dataset data/fea --config research/fea_prediction/configs/paper_fea.json
python research/cfd_prediction/study.py init --name cfd_reproduction
python research/cfd_prediction/study.py run --name cfd_reproduction
python research/geometry_validation/main.py init --study geometry_reproduction
python research/geometry_validation/main.py run --study geometry_reproduction
```

For other dataset locations use `--dataset` for FEA training, `LOGIC_CFD_DATA` for CFD training and `LOGIC_FEA_DATA` for geometry validation. The geometry experiment uses the original IDs only in its deterministic selection of the 50 reused designs. Fresh runs report public FEA IDs. Exact subset membership for the mechanical study is distributed in `results/fea/subsets.json`.

## Source layout

| Path | Purpose |
|---|---|
| `tpms_hybrid.py`, `orientation_mapping.py`, `examples/` | Geometry generation and orientation-field integration |
| `research/geometry_validation/` | Control-field experiments and initial-geometry measurements |
| `research/fea_prediction/` | FEA encoders, POD reconstruction, training, evaluation and tests |
| `research/cfd_prediction/` | Conditioned residual 3D U-Net, field and hydraulic readouts, training and evaluation |
| `dataset_fea/` | Native geometry, meshing, Abaqus/Explicit simulation, label export and batch processing |
| `dataset_cfd/` | Fluid-domain construction, remeshing, COMSOL batch solving and field export |

## Native simulation dependencies

The released labels and trained models can be used directly with Python. Regenerating native solutions also requires the appropriate Abaqus/Explicit, Materialise 3-matic, Blender and COMSOL installations. The COMSOL Python bridge used MPh 1.3.1. The fixed CFD template is an external native input with SHA-256 `9fc8d1273435eb93948d76261d6180a14da690631f29315b039678da3e452857`. Set `LOGIC_COMSOL_TEMPLATE` to that file. Set `LOGIC_ABAQUS`, `LOGIC_TRIMATIC` and `LOGIC_BLENDER` for local executable paths. Historical configuration records retain original file references to preserve their checksums.

The CFD batch stages are `1_batch_fluid.py`, `2_batch_cfd.py` and `3_collect_results.py`. A new native campaign requires a source inventory referencing its prepared geometries and meshes. The inventory builder is in `dataset_cfd/campaign/build_inventory.py`. The release distributes initial design arrays, sampled response fields and scalar labels. Native solver working directories and job databases are outside this distribution.

## Citation

- Cite the archived software using the metadata in [CITATION.cff](CITATION.cff). [Software DOI](https://doi.org/10.5281/zenodo.24410073)
- Cite the simulation data, splits, trained weights and reference predictions using the companion dataset DOI. [Dataset DOI](https://doi.org/10.57967/hf/11735) · [Dataset snapshot for the paper](https://huggingface.co/datasets/pxy1118/LoGIC-Dataset/tree/97988689ed31266beffc05f4af5b46280bd62795)

Source code uses the accompanying MIT license.
