#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
from pydantic import BaseModel

class FileMap(BaseModel):
    """File mapping schema for aerodynamic datasets.

    Maps field names to their corresponding file names in the dataset directory.
    This allows different datasets to use different file naming conventions while maintaining a unified interface.
    """

    # Surface field files
    surface_position: str
    surface_pressure: str
    surface_friction: str

    # Volume field files
    volume_position: str
    volume_pressure: str
    volume_velocity: str
    volume_vorticity: str

    # Optional additional surface position files (dataset-specific)
    surface_position_stl: str | None = None
    surface_position_stl_resampled: str | None = None

    # Optional volume friction
    volume_friction: str | None = None

    # Optional volume distance field
    volume_distance_to_surface: str | None = None

    # Optional design parameters file
    design_parameters: str | None = None

wing_filemap_parametric = FileMap(
    surface_position="surface_position.pt",
    surface_pressure="surface_pressure.pt",
    surface_friction="surface_wall_shear_stress.pt",
    volume_position="volume_position.pt",
    volume_pressure="volume_pressure.pt",
    volume_velocity="volume_velocity.pt",
    volume_vorticity="volume_vorticity.pt",
    design_parameters="design_parameters.pt",
)