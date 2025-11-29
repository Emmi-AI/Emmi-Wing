#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
from pathlib import Path

import pandas as pd
import torch


class AeroDataset:
    """Dataset implementation for aerodynamic datasets with volume and surface fields.

    This unified dataset class provides an interface for aerodynamics dataset with volume and surface fields.
    The dataset behavior such as the dataset choice, train/val/test split IDs, etc.
    is configured through constructor parameters, allowing for easy extension to new datasets.

    Args:
        split: Which split to use, e.g. "train", "val", "test".
            Must be one of the splits defined in the DatasetSplitIDs feteched from DATASPLIT_REGISTRY.
            Can also include subset specifications using slice notation, e.g.:
            - "train[0]" for just the first training sample
            - "test[:50]" for the first 50 test samples
            - "val[10:20]" for validation samples 10-19
        filemap_name: Name of the FileMap object to fetch from FILEMAP_REGISTRY.
            FileMap contains the mapping from field names to file names.
        datasplit_name: Name of the DatasetSplitIDs object to fetch from DATASPLIT_REGISTRY.
            DatasetSplitIDs contains the simulation indices for the respective training and evaluation splits.
        root: path to the processed dataset, e.g. /nfs-gpu/research/datasets/drivaerml/preprocessed/subsampled_10x.
        num_repeats: Number of times to repeat the dataset (default 1).
        faulty_runs_file: a blacklist .csv file specifying run indicies that are faulty and to be ignored.
    """

    def __init__(
        self,
        split: str,
        root: str | None = None,
        num_repeats: int = 1,
        faulty_runs_file: str | None = None,
        **kwargs,
    ):
        self.faulty_runs_file = faulty_runs_file
        self.faulty_runs = None
        self.root = root
        if self.faulty_runs_file is not None:
            self.faulty_runs = self._parse_faulty_runs_csv()

        # Look up the objects from the registries
        self.filemap = kwargs["filemap"]
        self.design_parameters = kwargs["design_parameters"]
        self.datasplits = {k: list(v) for k, v in kwargs["datasplits"].model_dump().items()}
        self.num_repeats = num_repeats

        # Parse the split parameter and extract any subset specification
        self.split, subset_indices = self._parse_split_subset(split)

        if self.faulty_runs is not None:
            for k, v in self.datasplits.items():
                self.datasplits[k] = self._remove_faulty_runs(v)

        if self.split not in self.datasplits:
            raise ValueError(f"Unknown split '{self.split}'. Available splits: {list(self.datasplits.keys())}")

        # Apply subsetting if specified, otherwise use the full split
        all_design_ids = self.datasplits[self.split]
        self.design_ids = [all_design_ids[i] for i in subset_indices] if subset_indices else all_design_ids

    def _parse_faulty_runs_csv(self) -> set[int]:
        """Parse a CSV file containing faulty run IDs to be excluded from the dataset.

        The CSV file should contain a column named 'RUN_IDs' with integer values.

        Returns:
            set[int]: A set of faulty run IDs.
        """

        self.faulty_runs_file = Path(self.faulty_runs_file).expanduser().resolve()
        if not self.faulty_runs_file.exists():
            raise FileNotFoundError(f"Faulty runs file not found: {self.faulty_runs_file}")

        faulty_runs_df = pd.read_csv(self.faulty_runs_file)
        if "RUN_IDs" not in faulty_runs_df.columns:
            raise ValueError(
                f"'RUN_IDs' column not found in {self.faulty_runs_file}. Please ensure the CSV file has a 'RUN_IDs' column."
            )
        return set(faulty_runs_df["RUN_IDs"].astype(int).tolist())

    def _remove_faulty_runs(self, run_ids: list[int]) -> list[int]:
        """Remove faulty run IDs from the provided list based on the faulty_runs set.

        Args:
            run_ids (set[int]): set of run IDs to filter.

        Returns:
            list[int]: Filtered set of run IDs without faulty ones.
        """
        if self.faulty_runs is None:
            return run_ids
        return list(set(run_ids) - self.faulty_runs)

    def _parse_split_subset(self, split_spec: str) -> tuple[str, list[int] | range | None]:
        """Parse a split specification with optional subset info in brackets.

        Examples:
            "train" -> ("train", None)  # No subset, use full split
            "train[0]" -> ("train", [0])  # Single sample
            "test[:50]" -> ("test", range(0, 50))  # First 50 samples

        Returns:
            Tuple of (split_name, subset_indices) where subset_indices is:
            - None: when no brackets are used, indicating full split should be used
            - List of indices or range object: when brackets are used
        """
        # No brackets means no subset specification - return None for subset_indices
        if not ("[" in split_spec and split_spec.endswith("]")):
            return split_spec, None

        # Get split name and subset expression
        split_name, subset_expr = split_spec.split("[", 1)
        subset_expr = subset_expr.rstrip("]")

        try:
            # Single index case: "train[0]"
            if ":" not in subset_expr:
                return split_name, [int(subset_expr)]

            # Slice notation: convert "10:20" → range(10, 20)
            parts = [int(p) if p else None for p in subset_expr.split(":")]
            if len(parts) == 2:  # start:end format
                start = parts[0] or 0
                end = parts[1] or 1000000
                return split_name, range(start, end)
            elif len(parts) == 3:  # start:end:step format
                start = parts[0] or 0
                end = parts[1] or 1000000
                step = parts[2] or 1
                return split_name, range(start, end, step)
            else:
                raise ValueError("Invalid slice format")
        except ValueError as err:
            raise ValueError(f"Invalid subset expression: '{subset_expr}'") from err

    def __len__(self):
        return len(self.design_ids) * self.num_repeats

    def _load_from_disk(self, idx: int, filename: str) -> torch.Tensor:
        design_uri = self.root / f"run_{self.design_ids[idx]}"
        assert design_uri.exists(), f"{design_uri.as_posix()} does not exist"
        return torch.load(design_uri / filename, weights_only=True)

    def _load(self, idx: int, filename: str) -> torch.Tensor:
        wrapped_idx = idx % len(self.design_ids)  # due to repeating the dataset
        return self._load_from_disk(idx=wrapped_idx, filename=filename)

    def getitem_surface_position(self, idx: int) -> torch.Tensor:
        """Retrieves surface positions (num_surface_points, 3)"""
        return self._load(idx=idx, filename=self.filemap.surface_position)

    def getitem_surface_position_stl(self, idx: int) -> torch.Tensor:
        """Retrieves surface positions (num_surface_points, 3)"""
        if self.filemap.surface_position_stl is None:
            raise ValueError("surface_position_stl not available for this dataset")
        return self._load(idx=idx, filename=self.filemap.surface_position_stl)

    def getitem_surface_position_stl_resampled(self, idx: int) -> torch.Tensor:
        """Retrieves surface positions (num_surface_points, 3)"""
        if self.filemap.surface_position_stl_resampled is None:
            raise ValueError("surface_position_stl_resampled not available for this dataset")
        return self._load(idx=idx, filename=self.filemap.surface_position_stl_resampled)

    def getitem_surface_pressure(self, idx: int) -> torch.Tensor:
        """Retrieves surface pressures (num_surface_points, 1)"""
        return self._load(idx=idx, filename=self.filemap.surface_pressure).unsqueeze(1)

    def getitem_surface_friction(self, idx: int) -> torch.Tensor:
        """Retrieves surface friction (=wallshearstress) (num_surface_points, 3)"""
        return self._load(idx=idx, filename=self.filemap.surface_friction)

    def getitem_volume_position(self, idx: int) -> torch.Tensor:
        """Retrieves volume position (num_volume_points, 3)"""
        return self._load(idx=idx, filename=self.filemap.volume_position)

    def getitem_volume_pressure(self, idx: int) -> torch.Tensor:
        """Retrieves volume pressures (num_volume_points, 1)"""
        return self._load(idx=idx, filename=self.filemap.volume_pressure).unsqueeze(1)

    def getitem_volume_velocity(self, idx: int) -> torch.Tensor:
        """Retrieves volume velocity (num_volume_points, 3)"""
        return self._load(idx=idx, filename=self.filemap.volume_velocity)

    def getitem_volume_vorticity(self, idx: int) -> torch.Tensor:
        """Retrieves volume vorticity (num_volume_points, 3)"""
        return self._load(idx=idx, filename=self.filemap.volume_vorticity)

    def getitem_volume_distance_to_surface(self, idx: int) -> torch.Tensor:
        """Retrieves distance to surface of volume points (num_volume_points, 1)"""
        if self.filemap.volume_distance_to_surface is None:
            raise ValueError("volume_distance_to_surface not available for this dataset")
        return self._load(idx=idx, filename=self.filemap.volume_distance_to_surface).unsqueeze(1)

    def getitem_design_parameters(self, idx: int) -> torch.Tensor:
        """Retrieves design parameters for the simulation"""
        design_parameters = self._load_design_parameters_map(idx)
        return torch.stack(list(design_parameters.values()))

    def getitem_geometry_design_parameters(self, idx: int) -> torch.Tensor:
        """Retrieves geometry design parameters as a single tensor.

        Returns:
            torch.Tensor: Geometry design parameters tensor of shape (1, num_geometry_parameters)
        """
        if self.design_parameters is None:
            raise ValueError("design_parameters_name not provided. Unable to extract geometry parameters.")

        params_dict = self._load_design_parameters_map(idx)
        design_parameters = self.design_parameters(**params_dict)
        geometry_params = design_parameters.get_geometry_parameters_tensor()
        # unsqueeze for collator normalizer compatibility
        return geometry_params.unsqueeze(0)

    def getitem_inflow_design_parameters(self, idx: int) -> torch.Tensor:
        """Retrieves inflow design parameters as a single tensor.

        Returns:
            torch.Tensor: Inflow design parameters tensor of shape (1, num_inflow_parameters)
        """
        if self.design_parameters is None:
            raise ValueError("design_parameters_name not provided. Unable to extract inflow parameters.")

        params_dict = self._load_design_parameters_map(idx)
        design_parameters = self.design_parameters(**params_dict)
        inflow_params = design_parameters.get_inflow_parameters_tensor()
        # unsqueeze for collator normalizer compatibility
        return inflow_params.unsqueeze(0)

    def sample_info(self, idx: int) -> dict[str, str | int | None]:
        """Get information about a sample such as its local path, run name, etc."""
        idx = idx % len(self.design_ids)
        design_id = self.design_ids[idx]
        run_name = f"run_{design_id}"
        sample_uri = self.root / run_name
        return {
            "sample_uri": sample_uri,
            "run_name": run_name,
            "run_number": design_id,
            "design_id": design_id,
            "split": self.split,
        }

    def _load_design_parameters_map(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieves all design parameters as dictionary {<param_name>: <param_value>}.
        Assumes that the design parameters are stored as a dictionary in the file."""
        if self.filemap.design_parameters is None:
            raise ValueError("design_parameters not available for this dataset")
        return self._load(idx=idx, filename=self.filemap.design_parameters)


class WingDataset(AeroDataset):
    """Convenience AeroDataset class with default parameters for Wing dataset."""

    def __init__(
        self,
        split: str,
        filemap: str,
        datasplits: str,
        root: str | None,
        design_parameters: str = "wing",
        num_repeats: int = 1,
        faulty_runs_file: str | None = None,
        **kwargs,
    ):
        super().__init__(
            split=split,
            filemap=filemap,
            datasplits=datasplits,
            design_parameters=design_parameters,
            root=root,
            num_repeats=num_repeats,
            faulty_runs_file=faulty_runs_file,
            **kwargs,
        )


