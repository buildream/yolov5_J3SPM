
# Patched hubconf for PyTorch >= 2.6 (safe unpickling + weights_only)
# - Backward-compatible with older YOLOv5 repos (no C2f, etc.)
# - Registers safe globals for whatever modules exist in models.common / models.yolo
# - Forces torch.load(weights_only=False) (TRUSTED CHECKPOINTS ONLY)
# - Exposes `custom` entrypoint similar to upstream

import importlib
import torch
import torch.nn as nn
from torch.serialization import add_safe_globals, safe_globals

# ---- SECURITY NOTE ----
# This sets weights_only=False for torch.load within this process.
# Only use with checkpoints you trust!
_orig_load = torch.load
def _load_with_wo_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _load_with_wo_false
# -----------------------

# Dynamically collect available classes from models.common / models.yolo
_allow = [nn.Sequential, nn.ModuleList, nn.ModuleDict]

def _maybe_from(module_name, names):
    try:
        m = importlib.import_module(module_name)
    except Exception:
        return
    for n in names:
        obj = getattr(m, n, None)
        if obj is not None:
            _allow.append(obj)

# Common YOLOv5 blocks (some may not exist in older trees)
_m_common = [
    "Conv","Bottleneck","BottleneckCSP","C3","C2","C2f","SPPF","SPP","Focus","Concat","DetectMultiBackend"
]
_m_yolo = [
    "DetectionModel","SegmentationModel","ClassificationModel","Model","Detect"
]

_maybe_from("models.common", _m_common)
_maybe_from("models.yolo", _m_yolo)

# Register once
add_safe_globals(_allow)

# Import bits we need (guarded)
from pathlib import Path
try:
    from models.common import AutoShape  # not in very old trees
except Exception:
    AutoShape = None

try:
    from models.experimental import attempt_load
except Exception as e:
    raise RuntimeError("hubconf_patched: cannot import models.experimental.attempt_load") from e

# Utils (fallback if paths differ)
try:
    from utils.torch_utils import select_device
    from utils.general import LOGGER, ROOT, check_requirements, intersect_dicts, logging
except Exception:
    # minimal fallbacks
    def select_device(device=None): return torch.device("cpu")
    class _L: pass
    LOGGER = _L(); import logging as _pylog; LOGGER.setLevel = _pylog.getLogger("y5").setLevel
    import pathlib as _pl; ROOT = _pl.Path(__file__).parent
    def check_requirements(*a, **k): pass
    def intersect_dicts(a, b, exclude=()): return {k:v for k,v in a.items() if k in b and not any(e in k for e in exclude)}
    import logging

# Try to import DetectionModel class for non-pretrained path (optional)
try:
    from models.yolo import DetectionModel as _DetModel
except Exception:
    _DetModel = None

# Re-export official constructors from the original file for compatibility if present
try:
    from hubconf_orig import yolov5n, yolov5s, yolov5m, yolov5l, yolov5x, \
                             yolov5n6, yolov5s6, yolov5m6, yolov5l6, yolov5x6  # noqa: F401
except Exception:
    pass

def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    if not verbose:
        LOGGER.setLevel(logging.WARNING)

    try:
        check_requirements(ROOT / "requirements.txt", exclude=("opencv-python", "tensorboard", "thop"))
    except Exception:
        pass

    device = select_device(device)
    name = Path(name)
    path = name.with_suffix(".pt") if name.suffix == "" and not name.is_dir() else name

    try:
        # Custom checkpoint path (most common case)
        with safe_globals(_allow):
            model = attempt_load(path, device=device, fuse=False)

        # Optional autoshape (if available)
        if autoshape and AutoShape is not None:
            try:
                # Seg/Cls models may not support it; keep upstream warnings minimal
                model = AutoShape(model)
            except Exception:
                pass

        if not verbose:
            LOGGER.setLevel(logging.INFO)

        return model.to(device)

    except Exception as e:
        help_url = "https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading"
        s = f"{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help."
        raise Exception(s) from e

def custom(path="path/to/model.pt", autoshape=True, _verbose=True, device=None):
    return _create(path, pretrained=True, channels=3, classes=80, autoshape=autoshape, verbose=_verbose, device=device)
