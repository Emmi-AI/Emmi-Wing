# Emmi-Wing

This repository contains the dataset proposed in [Going with the Speed of Sound: Pushing Neural Surrogates into Highly-turbulent Transonic Regimes](https://arxiv.org/abs/2511.21474), presented at the Workshop on ML for the Physical Sciences at NeurIPS 2025.

The full dataset can be downloaded at this link: https://data.emmi.ai/s/qTgKFQCRNnFTgXN

## Example Usage

We provide code to load the data and compute the drag and lift coefficients as shown in the paper.
The required code can be found in the `src` directory which is structured as follows:

    src
    ├── create_splits.py
    ├── datasets.py
    ├── datasplits.py
    ├── design_parameters.py
    ├── filemaps.py
    └── inspect_data.py

The provided scripts in this directory provide a simple API for loading and handling the dataset which can later be used for training neural surrogates. For now, we just use this simple API for visualizing the characteristics of the dataset. 

First, please either download the entire dataset from the link above, or pull a subset of the parameter scans from the [huggingface repository](https://huggingface.co/datasets/EmmiAI/Emmi-Wing) and place it in the `data` directory and unzip it.

To compute the drag and lift coefficients and visualize their distribution, simply execute:

    python inspect_data.py

This will load all the runs that are stored in the `data` directory in parallel and compute their associated drag and lift coefficients and finally generate some plots of those.
In case you experience deadlocks when executing this script, you can also execute it in sequential mode by adding the `--debug` option.
The `inspect_data.py` script also supports setting different datasplits (`all`, `scans`, `scans_reduced`).
By default it is set to `scans_reduced` which corresponds to the data in the huggingface repo.

The datasplits specified in `datasplits.py` are the ones used for the experiments in the paper and can be reproduced by executing the following command

    python create_splits.py 

## OpenFOAM Template Setup

The simulation files are located in the `Template` directory. To prepare the case for execution, follow the steps below to import the geometry and configure the flow parameters.

### 1. Generate and Place Geometry
First, generate the wing geometry using the provided function:

    generate_wing_mesh(
        naca_digits,
        chord_root,
        span,
        taper_ratio,
        sweep,
        n_chord=640,
        n_span=380,
        close_root=True)

This will output an `.stl` file. You must place this file into the surface directory of the template:

`Template/constant/triSurface/`

### 2. Configure Mesh Dictionaries
You need to link the generated `.stl` file to the meshing dictionaries. Open the following two files:

1.  `Template/system/surfaceFeatureExtractDict`
2.  `Template/system/snappyHexMeshDict`

In both files, locate the placeholder string **`NAME_OF_CAD`** and replace it with the actual name of your `.stl` file (e.g., `wing.stl`).

### 3. Set Velocity Components
Finally, define the initial velocity boundary conditions. Open the velocity file:

`Template/0.orig/U`

Update the internal field and inlet values by prescribing the velocity components **$U_x$** and **$U_y$**.

## Citation

If you find our work useful or use our dataset, please consider citing it

    @inproceedings{
        paischer2025going,
        title={Going with the Speed of Sound: Pushing Neural Surrogates into Transonic and Highly Turbulent Regimes},
        author={Anonymous},
        booktitle={Machine Learning and the Physical Sciences Workshop @ NeurIPS 2025},
        year={2025},
        url={https://openreview.net/forum?id=36Tpmdy1Cu}
        }