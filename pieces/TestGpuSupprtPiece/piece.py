import json
import logging

from domino.base_piece import BasePiece

from .models import InputModel, OutputModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def inspect_pytorch_gpu():
    import torch

    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
            }
        )

    gpu_operation_result = None
    gpu_operation_error = None
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda:0")
            matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
            gpu_operation_result = torch.matmul(matrix, matrix).cpu().tolist()
        except Exception as exc:
            gpu_operation_error = f"{type(exc).__name__}: {exc}"

    return {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            str(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        ),
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "current_device": (
            torch.cuda.current_device() if torch.cuda.is_available() else None
        ),
        "gpu_available": torch.cuda.is_available() and torch.cuda.device_count() > 0,
        "gpu_operation_result": gpu_operation_result,
        "gpu_operation_error": gpu_operation_error,
    }


class TestGpuSupprtPiece(BasePiece):
    def piece_function(self, input_data: InputModel):
        report = inspect_pytorch_gpu()
        logger.info("PyTorch GPU probe result:\n%s", json.dumps(report, indent=2))
        return OutputModel(**report)
