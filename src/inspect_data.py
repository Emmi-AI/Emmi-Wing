#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""
Plot Wing OOD dataset splits - 2D scatter plot only.
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from datasets import WingDataset
from datasplits import WingParametricSplitIDs, WingParametricScanIDs, WingParametricReducedScanIDs, DatasetSplitIDs
from design_parameters import WingDesignParameters
from filemaps import wing_filemap_parametric

# Global plotting constants
COLORS = {
    "train": "royalblue",
    "val": "lightblue",
    "test": "lightgreen",
    "test_extrapol": "firebrick",
    "test_interpol": "darkorange",
}

SIZES = {"train": 15, "val": 30, "test": 30, "test_extrapol": 60, "test_interpol": 60}

# Plot in order so OOD points are on top
PLOT_ORDER = ["train", "val", "test", "test_extrapol", "test_interpol"]


def load_design_parameters(data_root: Path) -> dict[int, dict[str, float]]:
    """Load design parameters from all run directories."""
    design_params = {}

    run_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("run_")])
    print(f"Found {len(run_dirs)} run directories")

    for run_dir in run_dirs:
        run_id = int(run_dir.name.split("_")[1])
        design_params_file = run_dir / "design_parameters.pt"

        if not design_params_file.exists():
            raise ValueError(f"Warning: Design parameters file not found for {run_dir.name}")

        params_dict = torch.load(design_params_file, weights_only=True)
        design_params[run_id] = {k: float(v) for k, v in params_dict.items()}

    print(f"Loaded design parameters for {len(design_params)} runs")
    return design_params


def create_parameter_dataframe(
    design_params: dict[int, dict[str, float]],
    split_ids: DatasetSplitIDs,
) -> pd.DataFrame:
    """Create a pandas DataFrame with parameters and split assignments."""

    records = []
    for run_id, params in design_params.items():
        wing_params = WingDesignParameters(**params)

        if run_id in split_ids.train:
            split = "train"
        elif run_id in split_ids.val:
            split = "val"
        elif run_id in split_ids.test:
            split = "test"
        elif run_id in split_ids.test_extrapol:
            split = "test_extrapol"
        elif run_id in split_ids.test_interpol:
            split = "test_interpol"
        else:
            split = "unknown"
            print(f"Warning: Run ID {run_id} not found in any split")

        record = {"run_id": run_id, "split": split, **wing_params.to_dict()}
        records.append(record)

    df = pd.DataFrame(records)
    print(f"Created DataFrame with {len(df)} samples and {len(df.columns)} columns")
    return df

def _get_parameter_info(df: pd.DataFrame, parameter_type: str) -> tuple[list[str], list[str], str]:
    """Get parameter information and filter out constant parameters."""
    wing_params = WingDesignParameters(**df.iloc[0].to_dict())

    if parameter_type == "inflow":
        param_dict = wing_params.get_inflow_parameters()
        param_names = list(param_dict.keys())
        title_suffix = "Inflow Parameter Space"
    elif parameter_type == "geometry":
        param_dict = wing_params.get_geometry_parameters()
        param_names = list(param_dict.keys())
        title_suffix = "Geometry Parameter Space"
    else:
        param_dict = wing_params.to_dict()
        param_names = list(param_dict.keys())
        title_suffix = "Full Parameter Space"

    # Filter out constant parameters
    varying_params = []
    for param in param_names:
        if df[param].nunique() > 1:  # Parameter varies
            varying_params.append(param)

    return param_names, varying_params, title_suffix

def plot_coefficients(df, output_dir, x_key="aoa", y_key="inflow_velocity", c_key=None):
    if c_key is not None:
        plt.scatter(df[x_key], df[y_key], c=df[c_key], s=2)
        plt.colorbar(label=c_key)
    else:
        plt.scatter(df[x_key], df[y_key], s=2)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.tight_layout()
    if c_key is not None:
        plot_file = output_dir / f"{c_key}_vs_{x_key}_and_{y_key}.png"
    else:
        plot_file = output_dir / f"{x_key}_and_{y_key}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

def get_wing_directions(aoa):
    alpha = np.deg2rad(aoa)
    drag_dir = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    lift_dir = np.array([-np.sin(alpha), 0.0, np.cos(alpha)])
    span_dir = np.cross(lift_dir, drag_dir)
    return drag_dir, span_dir, lift_dir

def get_drag_lift_coeffs(data: dict, simulation_parameters: dict):
    # Define and load reference values
    rhoInfty = 1.225
    Uinfty = simulation_parameters["inflow_design_parameters"][0, 0].item()
    aoa = simulation_parameters["inflow_design_parameters"][0, 1].item()
    chord_root = simulation_parameters["geometry_design_parameters"][0, 0].item()
    span_length = simulation_parameters["geometry_design_parameters"][0, 1].item()
    taper_ratio = simulation_parameters["geometry_design_parameters"][0, 2].item()
    # trapezoid reference area
    c_t = chord_root * taper_ratio
    reference_area = span_length * (chord_root + c_t) / 2
    qInfty = 0.5 * rhoInfty * Uinfty**2 * reference_area

    points = data["surface_position"].numpy()
    pressure = data["surface_pressure"].numpy()
    friction = data["surface_friction"].numpy()
    root = os.path.join(simulation_parameters["root"], simulation_parameters["run_id"])
    cell_centers = np.load(os.path.join(root, "cell_centers.npy"))
    cell_normals = np.load(os.path.join(root, "cell_normals.npy"))
    cell_areas = np.load(os.path.join(root, "cell_areas.npy"))
    
    Fp = -pressure * (cell_normals * cell_areas[:, None])
    Ff = (friction* cell_areas[:, None]).sum(0)
    total_force = Fp.sum(0) + Ff
    drag_dir, span_dir, lift_dir = get_wing_directions(aoa)

    # calculate coefficients
    target_cd = np.dot(total_force, -drag_dir) / qInfty
    target_cl = np.dot(total_force, -lift_dir) / qInfty
    target_cs = np.dot(total_force, -span_dir) / qInfty

    return {
            "drag_coeff": target_cd.item(), 
            "lift_coeff": target_cl.item(), 
            "side_force_coeff": target_cs.item(),
            "AoA": aoa,
            "U_infty": Uinfty,
            }

def collect_coefficients(run_id, dataset, run_name_idx_map):
    # compute drag/lift coefficients for datasamples
    cur_idx = run_name_idx_map[run_id]

    raw_sample = {
        "surface_position": dataset.getitem_surface_position(cur_idx),
        "surface_pressure": dataset.getitem_surface_pressure(cur_idx),
        "surface_friction": dataset.getitem_surface_friction(cur_idx),
        "geometry_design_parameters": dataset.getitem_geometry_design_parameters(cur_idx),
        "inflow_design_parameters": dataset.getitem_inflow_design_parameters(cur_idx),
    }

    simulation_parameters = {
        "geometry_design_parameters": raw_sample["geometry_design_parameters"],
        "inflow_design_parameters": raw_sample["inflow_design_parameters"],
        "run_id": f"run_{run_id}",
        "root": dataset.root,
    }

    coeffs = get_drag_lift_coeffs(raw_sample, simulation_parameters)
    coeffs["run_id"] = run_id
    return coeffs


@click.command()
@click.option(
    "--data-root",
    default="../data/",
    help="Path to preprocessed Wing dataset",
)
@click.option("--output-dir", default="./plots", help="Directory to save plot")
@click.option(
    "--datasplit",
    type=click.Choice(["all", "scans", "scans_reduced"]),
    default="scans_reduced",
    help="Which dataset to plot",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Execute script in debugging mode (w/o multiprocessing)",
)
def main(
    data_root: str, output_dir: str, datasplit: str, debug: bool
):
    """Plot Wing OOD dataset splits - creates all possible plots (1D, 2D, 3D) based on parameter dimensionality."""

    data_root_path = Path(data_root)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    if not data_root_path.exists():
        print(f"Error: Data root does not exist: {data_root_path}")
        sys.exit(1)

    print(f"Loading split IDs for {datasplit}...")
    if datasplit == "all":
        split_ids = WingParametricSplitIDs()
    elif datasplit == "scans":
        split_ids = WingParametricScanIDs()
    else:
        split_ids = WingParametricReducedScanIDs()

    print("Loading design parameters...")
    design_params = load_design_parameters(data_root_path)

    if len(design_params) == 0:
        print("Error: No design parameters found")
        sys.exit(1)

    print("Creating parameter DataFrame...")
    df = create_parameter_dataframe(design_params, split_ids)
    design_params = WingDesignParameters(**df.iloc[0].to_dict())
    coeff_dict = {}
    splits = [s for s in ["train", "val", "test", "test_interpol", "test_extrapol"] if len(getattr(split_ids, s))]
    for split in splits:
        dataset = WingDataset(
            split=split,
            filemap=wing_filemap_parametric,
            datasplits=split_ids,
            design_parameters=WingDesignParameters,
            root=data_root_path,
        )
        run_name_idx_map = {dataset.sample_info(idx)["run_number"]: idx for idx in range(len(dataset))}
        if not debug:
            executor_fn = partial(collect_coefficients, dataset=dataset, run_name_idx_map=run_name_idx_map)
            args = list(run_name_idx_map.keys())

            with ThreadPoolExecutor(32) as executor:
                # indices in parallel, collect results in list
                returns = list(
                    tqdm(
                        executor.map(executor_fn, args),
                        total=len(args),
                        desc="Computing drag/lift coefficients...",
                    )
                )

            for res in returns:
                coeff_dict[res["run_id"]] = res
        else:
            for run_id in tqdm(run_name_idx_map.keys(), desc="Computing coefficients...", total=len(run_name_idx_map)):
                coeffs = collect_coefficients(run_id, dataset, run_name_idx_map)
                coeff_dict[run_id] = coeffs
    
    ordered_coeffs = defaultdict(list)
    run_ids = df["run_id"].values
    for run_id in run_ids:
        if run_id in coeff_dict:
            for k, v in coeff_dict[run_id].items():
                ordered_coeffs[k].append(v)

    for k in ["drag_coeff", "lift_coeff", "side_force_coeff"]:
        df[k] = ordered_coeffs[k]
    
    plot_coefficients(df, output_path, x_key="aoa", y_key="lift_coeff", c_key=None)
    plot_coefficients(df, output_path, x_key="aoa", y_key="drag_coeff", c_key=None)
    plot_coefficients(df, output_path, x_key="drag_coeff", y_key="lift_coeff", c_key=None)

    plot_coefficients(df, output_path, x_key="aoa", y_key="inflow_velocity", c_key="drag_coeff")
    plot_coefficients(df, output_path, x_key="aoa", y_key="inflow_velocity", c_key="lift_coeff")
    plot_coefficients(df, output_path, x_key="aoa", y_key="inflow_velocity", c_key="side_force_coeff")

    print(df)
    df.to_csv(f"../data/{datasplit}_params_and_coeffs.csv")

if __name__ == "__main__":
    main()
