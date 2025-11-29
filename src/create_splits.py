#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Create OOD dataset splits for Wing dataset using convex hull peeling."""

import logging
import sys
from pathlib import Path

import click
import numpy as np
import torch
from scipy.spatial import ConvexHull

from design_parameters import WingDesignParameters


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def load_design_parameters(data_root: Path) -> dict[int, dict[str, float]]:
    logger = logging.getLogger(__name__)
    design_params = {}

    run_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("run_")])
    logger.info(f"Found {len(run_dirs)} run directories")

    for run_dir in run_dirs:
        run_id = int(run_dir.name.split("_")[1])
        design_params_file = run_dir / "design_parameters.pt"

        if not design_params_file.exists():
            raise ValueError(f"Warning: Design parameters file not found for {run_dir.name}")

        params_dict = torch.load(f=design_params_file, weights_only=True)
        design_params[run_id] = {k: float(v) for k, v in params_dict.items()}

    logger.info(f"Loaded design parameters for {len(design_params)} runs")
    return design_params


def extract_parameter_matrix(
    design_params: dict[int, dict[str, float]],
) -> tuple[np.ndarray, list[int], list[str]]:
    logger = logging.getLogger(__name__)

    sample_params = next(iter(design_params.values()))
    wing_params = WingDesignParameters(**sample_params)

    param_dict = wing_params.to_dict()
    param_names = list(param_dict.keys())
    logger.info(f"Using all parameters: {param_names}")

    run_ids = sorted(design_params.keys())
    param_matrix = np.zeros((len(run_ids), len(param_names)))

    for i, run_id in enumerate(run_ids):
        params = design_params[run_id]
        wing_params = WingDesignParameters(**params)
        param_values = list(wing_params.to_dict().values())
        param_matrix[i] = param_values

    logger.info(f"Parameter matrix shape: {param_matrix.shape}")
    for i, name in enumerate(param_names):
        min_val, max_val = param_matrix[:, i].min(), param_matrix[:, i].max()
        range_val = max_val - min_val
        logger.info(f"  {name}: [{min_val:.4f}, {max_val:.4f}] (range: {range_val:.6f})")

    # Check if parameters actually vary and remove constant dimensions
    param_ranges = param_matrix.max(axis=0) - param_matrix.min(axis=0)
    constant_params = param_ranges < 1e-10
    varying_params = ~constant_params

    if np.any(constant_params):
        constant_param_names = [param_names[i] for i in range(len(param_names)) if constant_params[i]]
        logger.warning(f"Constant parameters detected: {constant_param_names}")
        logger.info("Removing constant dimensions...")

        # Filter out constant dimensions
        param_matrix = param_matrix[:, varying_params]
        param_names = [param_names[i] for i in range(len(param_names)) if varying_params[i]]
        logger.info(f"Using {len(param_names)} varying parameters: {param_names}")

    if np.all(constant_params):
        logger.error(f"All parameters are constant across samples!")
        logger.error("Please check the data root path and parameter type (perhaps geometry and inflow are mixed).")
        sys.exit(1)

    return param_matrix, run_ids, param_names


def convex_hull_peeling(
    points: np.ndarray, m_extrapol: int, m_interpol: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger = logging.getLogger(__name__)

    if len(points) < m_extrapol + m_interpol:
        raise ValueError("Not enough points for requested split sizes")

    logger.info("--- Stage 1: Peeling for Extrapolation OOD Set ---")
    remaining_points = points.copy()
    remaining_indices = np.arange(len(points))
    extrap_indices = []

    logger.info(f"Starting with {len(remaining_points)} points in {remaining_points.shape[1]}D space")

    while len(extrap_indices) < m_extrapol and len(remaining_points) > 2:
        try:
            hull = ConvexHull(points=remaining_points, qhull_options="QJ")
            hull_vertices = hull.vertices
            original_indices = remaining_indices[hull_vertices]
            extrap_indices.extend(original_indices)

            logger.info(
                f"Peeled layer with {len(original_indices)} points. Total OOD Extrapolation points found: {len(extrap_indices)}"
            )

            remaining_points = np.delete(arr=remaining_points, obj=hull_vertices, axis=0)
            remaining_indices = np.delete(arr=remaining_indices, obj=hull_vertices, axis=0)
        except Exception as e:
            logger.warning(f"ConvexHull failed: {e}")
            logger.warning(f"Remaining points shape: {remaining_points.shape}")
            break

    final_extrap_indices = np.array(extrap_indices[:m_extrapol])

    logger.info("--- Stage 2: Peeling for Interpolation OOD Set ---")
    points_inward = remaining_points.copy()
    indices_inward = remaining_indices.copy()

    logger.info(f"Starting interpolation stage with {len(points_inward)} points, target: {m_interpol}")

    while len(points_inward) > m_interpol and len(points_inward) > 2:
        try:
            hull = ConvexHull(points=points_inward, qhull_options="QJ")
            hull_vertices = hull.vertices

            if len(points_inward) - len(hull_vertices) < m_interpol:
                logger.info(
                    f"Final boundary layer identified. Current core size: {len(points_inward)}, target: {m_interpol}."
                )
                num_to_peel = len(points_inward) - m_interpol
                vertices_to_peel = hull_vertices[:num_to_peel]
                logger.info(f"Peeling {num_to_peel} points from the final layer to achieve exact size.")

                points_inward = np.delete(arr=points_inward, obj=vertices_to_peel, axis=0)
                indices_inward = np.delete(arr=indices_inward, obj=vertices_to_peel, axis=0)
                break

            logger.info(
                f"Peeling inner layer of {len(hull_vertices)} points. {len(points_inward) - len(hull_vertices)} points remain."
            )
            points_inward = np.delete(arr=points_inward, obj=hull_vertices, axis=0)
            indices_inward = np.delete(arr=indices_inward, obj=hull_vertices, axis=0)
        except Exception:
            break

    final_interp_indices = indices_inward

    extrap_mask = np.isin(np.arange(len(points)), final_extrap_indices)
    interp_mask = np.isin(np.arange(len(points)), final_interp_indices)
    id_mask = ~extrap_mask & ~interp_mask
    final_id_indices = np.where(id_mask)[0]

    return final_id_indices, final_extrap_indices, final_interp_indices


def create_split_class(
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
    extrap_ids: list[int],
    interp_ids: list[int],
    train_run_ids_for_eval: list[int],
    class_name: str,
) -> str:
    test_set_str = "{" + ", ".join(map(str, sorted(test_ids))) + "}" if test_ids else "set()"
    num_samples_total = len(train_ids) + len(val_ids) + len(test_ids) + len(extrap_ids) + len(interp_ids)
    return f'''class {class_name}(DatasetSplitIDs):
    """Wing dataset split with OOD interpolation and extrapolation sets, {num_samples_total} all design parameter variations."""

    # fmt: off
    train: set[int] = {{{", ".join(map(str, sorted(train_ids)))}}}
    val: set[int] = {{{", ".join(map(str, sorted(val_ids)))}}}
    test: set[int] = {test_set_str}
    test_extrapol: set[int] = {{{", ".join(map(str, sorted(extrap_ids)))}}}
    test_interpol: set[int] = {{{", ".join(map(str, sorted(interp_ids)))}}}
    train_subset: set[int] = {{{", ".join(map(str, sorted(train_run_ids_for_eval)))}}}
    # fmt: on
'''


@click.command()
@click.option(
    "--data-root",
    default="data/",
    help="Path to preprocessed Wing dataset",
)
@click.option("--num-extrapol", default=1000, help="Number of extrapolation OOD samples")
@click.option("--num-interpol", default=1000, help="Number of interpolation OOD samples")
@click.option("--num-random-test", default=1000, help="Number of random in-distribution samples for testing")
@click.option("--num-random-val", default=1000, help="Number of random in-distribution samples for validation")
@click.option("--num-train-for-eval", default=1000, help="Number of samples from training set used for logging metrics")
@click.option("--class-name", default="WingParametricSplitIDs", help="Name for the generated split class")
@click.option("--seed", default=42, help="Random seed for reproducibility")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def main(
    data_root: str,
    num_extrapol: int,
    num_interpol: int,
    num_random_test: int,
    num_random_val: int,
    num_train_for_eval: int,
    class_name: str,
    seed: int,
    verbose: bool,
):
    """Create OOD dataset splits for Wing dataset using convex hull peeling."""
    setup_logging(verbose=verbose)
    logger = logging.getLogger(__name__)

    np.random.seed(seed=seed)

    data_root_path = Path(data_root)

    if not data_root_path.exists():
        logger.error(f"Data root does not exist: {data_root_path}")
        sys.exit(1)

    # Load design parameters
    logger.info("Loading design parameters...")
    design_params = load_design_parameters(data_root=data_root_path)

    if len(design_params) == 0:
        logger.error("No design parameters found")
        sys.exit(1)

    # Check that the total number of samples is more than the desired split sizes
    num_eval_samples = num_extrapol + num_interpol + num_random_test + num_random_val
    assert len(design_params) > num_eval_samples, f"Not enough samples for requested split sizes: {len(design_params)}."

    # Extract parameter matrix
    param_matrix, run_ids, _ = extract_parameter_matrix(design_params=design_params)

    logger.info(f"Performing convex hull peeling with {num_extrapol} extrapol & {num_interpol} interpol samples")
    in_dist_indices, extrapol_indices, interpol_indices = convex_hull_peeling(
        points=param_matrix,
        m_extrapol=num_extrapol,
        m_interpol=num_interpol,
    )

    in_dist_run_ids = [run_ids[i] for i in in_dist_indices]
    extrapol_run_ids = [run_ids[i] for i in extrapol_indices]
    interpol_run_ids = [run_ids[i] for i in interpol_indices]

    np.random.shuffle(in_dist_run_ids)
    val_run_ids = in_dist_run_ids[:num_random_val]
    test_run_ids = in_dist_run_ids[num_random_val : num_random_val + num_random_test]
    train_run_ids = in_dist_run_ids[num_random_val + num_random_test :]
    train_run_ids_for_eval = train_run_ids[:num_train_for_eval]

    n_total = len(train_run_ids) + len(val_run_ids) + len(test_run_ids) + len(extrapol_run_ids) + len(interpol_run_ids)
    assert n_total == len(design_params), (
        f"Total number of samples in splits ({n_total}) does not match number of design parameters ({len(design_params)})."
    )

    logger.info("\nFinal split sizes:")
    logger.info(f"  Train: {len(train_run_ids)} samples")
    logger.info(f"  Val: {len(val_run_ids)} samples")
    logger.info(f"  Test: {len(test_run_ids)} samples")
    logger.info(f"  Extrapolation OOD: {len(extrapol_run_ids)} samples")
    logger.info(f"  Interpolation OOD: {len(interpol_run_ids)} samples")
    logger.info(f"  Total: {n_total} samples.")

    class_definition = create_split_class(
        train_ids=train_run_ids,
        val_ids=val_run_ids,
        test_ids=test_run_ids,
        extrap_ids=extrapol_run_ids,
        interp_ids=interpol_run_ids,
        train_run_ids_for_eval=train_run_ids_for_eval,
        class_name=class_name,
    )

    print("\n" + "=" * 80)
    print("GENERATED CLASS DEFINITION:")
    print("=" * 80)
    print("from aero_cfd.datasets.datasplits import DatasetSplitIDs\n")
    print(class_definition)
    print("=" * 80)
    logger.info("OOD split creation completed successfully!")


if __name__ == "__main__":
    main()
