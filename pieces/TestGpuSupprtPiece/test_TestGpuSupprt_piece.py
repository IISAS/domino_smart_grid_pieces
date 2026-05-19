import importlib.util
import sys
import types
from pathlib import Path

_piece_dir = Path(__file__).resolve().parent
_pkg_name = "pieces.TestGpuSupprtPiece"

_models_spec = importlib.util.spec_from_file_location(
    f"{_pkg_name}.models", _piece_dir / "models.py"
)
_models = importlib.util.module_from_spec(_models_spec)
assert _models_spec and _models_spec.loader
_models_spec.loader.exec_module(_models)

_pkg = types.ModuleType(_pkg_name)
_pkg.models = _models
sys.modules.setdefault("pieces", types.ModuleType("pieces"))
sys.modules[_pkg_name] = _pkg
sys.modules[f"{_pkg_name}.models"] = _models

_piece_spec = importlib.util.spec_from_file_location(
    f"{_pkg_name}.piece", _piece_dir / "piece.py"
)
_piece = importlib.util.module_from_spec(_piece_spec)
assert _piece_spec and _piece_spec.loader
_piece_spec.loader.exec_module(_piece)
inspect_pytorch_gpu = _piece.inspect_pytorch_gpu


class _FakeCudaProps:
    name = "Fake GPU"
    total_memory = 16_000_000_000
    major = 8
    minor = 9
    multi_processor_count = 80


def test_inspect_pytorch_gpu_reports_devices(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.test"

    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda index: _FakeCudaProps(),
        current_device=lambda: 0,
    )
    fake_torch.cuda = fake_cuda
    fake_torch.version = types.SimpleNamespace(cuda="12.test")
    fake_torch.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(
            is_available=lambda: True,
            version=lambda: 90100,
        )
    )
    fake_torch.device = lambda name: name

    class _FakeTensor:
        def __init__(self, value, device=None):
            self._value = value

        def cpu(self):
            return self

        def tolist(self):
            return [[7.0, 10.0], [15.0, 22.0]]

    fake_torch.tensor = lambda value, device=None: _FakeTensor(value, device=device)
    fake_torch.matmul = lambda left, right: left

    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    report = inspect_pytorch_gpu()

    assert report["pytorch_version"] == "2.test"
    assert report["cuda_available"] is True
    assert report["cuda_version"] == "12.test"
    assert report["cudnn_version"] == "90100"
    assert report["gpu_available"] is True
    assert report["device_count"] == 1
    assert report["devices"][0]["name"] == "Fake GPU"
    assert report["current_device"] == 0
    assert report["gpu_operation_result"] == [[7.0, 10.0], [15.0, 22.0]]
    assert report["gpu_operation_error"] is None


def test_inspect_pytorch_gpu_no_cuda(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.test"
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        current_device=lambda: (_ for _ in ()).throw(RuntimeError("no cuda")),
    )
    fake_torch.version = types.SimpleNamespace(cuda=None)
    fake_torch.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(is_available=lambda: False, version=lambda: None)
    )
    fake_torch.device = lambda name: name
    fake_torch.tensor = lambda value, device=None: value
    fake_torch.matmul = lambda left, right: left

    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    report = inspect_pytorch_gpu()

    assert report["cuda_available"] is False
    assert report["gpu_available"] is False
    assert report["device_count"] == 0
    assert report["devices"] == []
    assert report["gpu_operation_result"] is None
    assert report["gpu_operation_error"] is None
