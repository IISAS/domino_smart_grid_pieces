from typing import Any

from pydantic import BaseModel, Field


class InputModel(BaseModel):
    """TestGpuSupprt Input Model"""


class OutputModel(BaseModel):
    """TestGpuSupprt Output Model"""

    pytorch_version: str = Field(description="Loaded PyTorch version.")
    cuda_available: bool = Field(
        description="Whether PyTorch can access CUDA at runtime."
    )
    cuda_version: str | None = Field(
        description="CUDA version PyTorch was built against."
    )
    cudnn_version: str | None = Field(description="cuDNN version when available.")
    device_count: int = Field(description="Number of CUDA devices visible to PyTorch.")
    devices: list[dict[str, Any]] = Field(
        description="CUDA device metadata reported by PyTorch."
    )
    current_device: int | None = Field(
        description="Current CUDA device index when a GPU is available."
    )
    gpu_available: bool = Field(
        description="True when PyTorch sees at least one CUDA device."
    )
    gpu_operation_result: list[list[float]] | None = Field(
        description="Result of a small PyTorch matmul on GPU when available."
    )
    gpu_operation_error: str | None = Field(
        description="Error from the small PyTorch GPU operation when it could not run."
    )
