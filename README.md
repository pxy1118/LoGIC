# LoGIC

[![DOI](https://zenodo.org/badge/1163288122.svg)](https://doi.org/10.5281/zenodo.20411608)

Official code repository for the manuscript **LoGIC-Design: A Low-Dimensional Grid-Controlled Framework for Heterogeneous TPMS Architectures and Data-Driven Performance Prediction**.

LoGIC-Design, short for Low-dimensional Grid-based Intent-Controlled Design, is a framework for generating heterogeneous triply periodic minimal surface (TPMS) architectures and predicting their mechanical and fluidic performance. The repository includes the core TPMS generator, automated finite element analysis (FEA) and computational fluid dynamics (CFD) dataset-generation pipelines, and deep learning surrogate models based on CNNs and GNNs.

## Features

- **Heterogeneous TPMS generation**: controls topology weights, density fields, and rotation fields through low-dimensional grids such as `3 x 3 x 3`, enabling smooth transitions across heterogeneous TPMS architectures and supporting both solid and fluid domain generation.
- **Automated dataset generation**:
  - **FEA pipeline**: batch geometry generation, meshing, Abaqus simulation, and post-processing for structural mechanics datasets.
  - **CFD pipeline**: fluid-domain extraction and processing with Blender, Materialise 3-matic, and COMSOL Multiphysics.
- **Deep learning performance prediction**:
  - **CNN model**: predicts structural responses from 3D voxel representations.
  - **GNN model**: predicts fluidic performance from pore-network graphs, with local features extracted by CNN backbones.
- **Interactive UI**: provides a Tkinter-based interface for interactive TPMS design and preview.

## Repository Structure

```text
LoGIC/
├── tpms_hybrid.py              # Core TPMS generation algorithm
├── UI/                         # Tkinter graphical user interface
│   └── tpms_ui.py              # UI entry point
├── dataset_generate_FEA/       # FEA dataset-generation pipeline
│   ├── 1_generate_dataset.py   # Generate TPMS samples
│   ├── 3_batch_run_abaqus.py   # Run Abaqus simulations in batch
│   └── ...
├── dataset_generate_CFD/       # CFD dataset-generation pipeline
│   ├── 1_generate_dataset.py   # Generate fluid-domain samples
│   ├── 2_process_fluid_blender.py
│   ├── 4_run_comsol_simulation_v2.py
│   └── ...
├── CNN/                        # CNN surrogate model
│   ├── configs/                # Model configuration files
│   ├── models/                 # CNN architectures
│   └── scripts/                # Training, prediction, and evaluation scripts
└── GNN/                        # Graph neural network surrogate model
    ├── configs/                # Model configuration files
    ├── models/                 # Pore-GNN architectures
    └── scripts/                # Graph construction, training, and evaluation scripts
```

## Dataset

The dataset associated with this project is hosted on Hugging Face:

[https://huggingface.co/datasets/pxy1118/LoGIC-Dataset](https://huggingface.co/datasets/pxy1118/LoGIC-Dataset)

Place downloaded files in the corresponding directories:

- `CNN/dataset/`
- `GNN/dataset/`
- `dataset_generate_CFD/` (requires `fluid.mph` for the CFD workflow)

## Version and DOI

This repository is archived through GitHub Releases and the Zenodo GitHub integration. The current public Zenodo DOI is:

- DOI: [`10.5281/zenodo.20411608`](https://doi.org/10.5281/zenodo.20411608)
- Archived version: `v1.0.0`

## Installation

### Python Environment

Python 3.11 or later is recommended. Install the required Python dependencies with:

```bash
pip install -r requirements.txt
```

If GPU acceleration is needed, install the PyTorch build that matches the local CUDA version.

### External Software Dependencies

The full dataset-generation workflow requires several commercial or external simulation tools:

- **Abaqus** for FEA simulation in `dataset_generate_FEA`.
- **Blender** for initial fluid-domain mesh processing in `dataset_generate_CFD`.
- **Materialise 3-matic** for fluid-domain remeshing and mesh optimization.
- **COMSOL Multiphysics** for CFD simulation.

These tools are only required for regenerating simulation datasets. The core generator and learning-model scripts can be used separately when the required data are available.

## Usage

### Interactive Design UI

Run the Tkinter interface for interactive TPMS design and preview:

```bash
python UI/tpms_ui.py
```

### FEA Dataset Generation

Run the numbered scripts in order:

```bash
cd dataset_generate_FEA
python 1_generate_dataset.py
python 2_process_volmesh.py
python 3_batch_run_abaqus.py
python 4_batch_postprocess.py
python 5_aggregate_postprocessed_curves.py
```

### CFD Dataset Generation

```bash
cd dataset_generate_CFD
python 1_generate_dataset.py
```

The following CFD steps call Blender, Materialise 3-matic, and COMSOL. See the inline notes in the corresponding scripts for local software-path configuration.

### CNN Training and Evaluation

```bash
cd CNN
python scripts/1_train_cnn.py
python scripts/2_predict.py
python scripts/3_evaluate_all.py
python scripts/4_visualize_results.py
```

### GNN Training and Evaluation

```bash
cd GNN
python scripts/1_pretrain_cnn.py
python scripts/2_extract_features.py
python scripts/3_build_graphs.py
python scripts/4_train_gnn.py
python scripts/5_predict.py
python scripts/6_evaluate_all.py
```

## Citation

If you use this codebase in your research, please cite the archived software release. GitHub reads the repository-level [`CITATION.cff`](CITATION.cff) file and displays a citation entry automatically. Please cite the Zenodo DOI for the archived `v1.0.0` release.

```bibtex
@software{pan_logic_design_2026,
  author  = {Pan, Xiaoyue},
  title   = {LoGIC-Design: A Low-Dimensional Grid-Controlled Framework for Heterogeneous TPMS Architectures and Data-Driven Performance Prediction},
  year    = {2026},
  version = {1.0.0},
  doi     = {10.5281/zenodo.20411608},
  url     = {https://github.com/pxy1118/LoGIC}
}
```

After the manuscript is formally published, please also cite the paper DOI or journal reference.

## License

This project is licensed under the [MIT License](LICENSE).
