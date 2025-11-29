#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
from abc import ABC, abstractmethod

import torch
from pydantic import BaseModel


class BaseDesignParameters(BaseModel, ABC):
    """Base class for design parameter models.

    Provides common functionality for converting parameters to tensors
    and splitting into geometry/inflow parameters.
    """

    @abstractmethod
    def get_geometry_parameters(self) -> dict[str, float]:
        """Get geometry-related design parameters.

        Returns:
            Dictionary of parameter names to values for geometry parameters
        """
        raise NotImplementedError("Must implement get_geometry_parameters()")

    @abstractmethod
    def get_inflow_parameters(self) -> dict[str, float]:
        """Get inflow-related design parameters.

        Returns:
            Dictionary of parameter names to values for inflow parameters
        """
        raise NotImplementedError("Must implement get_inflow_parameters()")

    def get_geometry_parameters_tensor(self) -> torch.Tensor:
        """Get geometry parameters as a tensor.

        Returns:
            torch.Tensor: Geometry parameters as a 1D tensor
        """
        params = self.get_geometry_parameters()
        return torch.tensor(list(params.values()), dtype=torch.float32)

    def get_inflow_parameters_tensor(self) -> torch.Tensor:
        """Get inflow parameters as a tensor.

        Returns:
            torch.Tensor: Inflow parameters as a 1D tensor
        """
        params = self.get_inflow_parameters()
        return torch.tensor(list(params.values()), dtype=torch.float32)

    def to_dict(self) -> dict[str, float]:
        """Convert all parameters to dictionary.

        Returns:
            Dictionary of all parameter names to values
        """
        return self.model_dump()

    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        """Convert all parameters to tensor dictionary.

        Returns:
            Dictionary of parameter names to tensor values
        """
        return {name: torch.tensor(value, dtype=torch.float32) for name, value in self.model_dump().items()}

class WingDesignParameters(BaseDesignParameters):
    """Design parameters for Wing dataset.

    Based on the actual parameter names from the Wing dataset CSV file.
    These parameters define the wing geometry and inflow conditions.
    """

    # Geometry parameters
    chord_root: float  # Root chord length
    span: float  # Wing span
    taper_ratio: float  # Wing taper ratio
    sweep: float  # Wing sweep angle
    dihedral: float  # Wing dihedral angle

    # Inflow parameters
    inflow_velocity: float  # Inflow velocity
    aoa: float  # Angle of attack

    def get_geometry_parameters(self) -> dict[str, float]:
        """Get geometry-related design parameters."""
        return {
            "chord_root": self.chord_root,
            "span": self.span,
            "taper_ratio": self.taper_ratio,
            "sweep": self.sweep,
            "dihedral": self.dihedral,
        }

    def get_inflow_parameters(self) -> dict[str, float]:
        """Get inflow-related design parameters."""
        return {
            "inflow_velocity": self.inflow_velocity,
            "aoa": self.aoa,
        }