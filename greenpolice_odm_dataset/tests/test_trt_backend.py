"""
Logic test for 07's TensorRTInferencer WITHOUT a GPU or TensorRT.

A fake `tensorrt` module emulates both the TensorRT >= 8.5 tensor API (TRT 10 style) and the
legacy binding API; the "engine" reads the input buffer through its raw data pointer and writes,
for every batch item b, one detection row whose confidence = mean(input[b]).  That checks what can
go wrong in the real backend: the Ultralytics metadata header, input/output discovery, static-batch
padding, dynamic-batch shape setting and output slicing, and per-item ordering.

It does NOT prove the class runs on a Jetson -- only real TensorRT can (07 --benchmark there).
Run:  python tests/test_trt_backend.py
"""
import contextlib
import ctypes
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T = 32           # tiny tile so the test is fast


# ───────────────────────────── fake tensorrt ─────────────────────────────────
def make_fake_trt(new_api: bool, dynamic: bool):
    trt = types.ModuleType("tensorrt")
    trt.__version__ = "10.3.0" if new_api else "8.4.1"

    class Logger:
        WARNING = 2
        def __init__(self, *a): pass
    trt.Logger = Logger

    class TensorIOMode:
        INPUT, OUTPUT = "in", "out"
    trt.TensorIOMode = TensorIOMode
    trt.nptype = lambda dt: dt                     # our fake dtypes ARE numpy types

    NAMES = ["images", "output0"]
    MAXB = 8 if dynamic else 4

    def in_shape():   # Ultralytics dynamic engines: batch AND spatial dims dynamic
        return (-1, 3, -1, -1) if dynamic else (MAXB, 3, T, T)

    def as_np(ptr, shape, dtype=np.float32):
        n = int(np.prod(shape))
        buf = (ctypes.c_float * n).from_address(ptr)
        return np.ctypeslib.as_array(buf).reshape(shape)

    class Context:
        def __init__(self):
            self.b = None if dynamic else MAXB
            self.addr = {}

        # new API
        def set_input_shape(self, name, shp):
            assert name == "images" and dynamic and tuple(shp[1:]) == (3, T, T), shp
            self.b = shp[0]
            return True

        def get_tensor_shape(self, name):
            b = self.b if self.b is not None else -1
            return (b, 3, T, T) if name == "images" else (b, 300, 6)

        def set_tensor_address(self, name, ptr):
            self.addr[name] = ptr

        def execute_async_v3(self, stream):
            return self._run(self.addr["images"], self.addr["output0"])

        # legacy API
        def set_binding_shape(self, idx, shp):
            assert idx == 0 and dynamic and tuple(shp[1:]) == (3, T, T), shp
            self.b = shp[0]
            return True

        def get_binding_shape(self, idx):
            return self.get_tensor_shape(NAMES[idx])

        def execute_async_v2(self, bindings, stream):
            return self._run(bindings[0], bindings[1])

        def _run(self, pin, pout):
            b = self.b
            x = as_np(pin, (b, 3, T, T))
            y = as_np(pout, (b, 300, 6))
            y[:] = 0
            for i in range(b):
                y[i, 0] = [T / 2, T / 2, 4, 4, float(x[i].mean()), i % 18]   # cx cy w h conf cls (px)
            return True

    class Engine:
        def create_execution_context(self):
            return Context()

    if new_api:
        Engine.num_io_tensors = 2
        Engine.get_tensor_name = lambda self, i: NAMES[i]
        Engine.get_tensor_mode = lambda self, n: TensorIOMode.INPUT if n == "images" else TensorIOMode.OUTPUT
        Engine.get_tensor_shape = lambda self, n: in_shape() if n == "images" else (in_shape()[0], 300, 6)
        Engine.get_tensor_dtype = lambda self, n: np.float32
        Engine.get_tensor_profile_shape = lambda self, n, p: ((1, 3, 32, 32), (4, 3, T, T), (MAXB, 3, 2 * T, 2 * T))
    else:
        Engine.num_bindings = 2
        Engine.get_binding_name = lambda self, i: NAMES[i]
        Engine.binding_is_input = lambda self, i: i == 0
        Engine.get_binding_index = lambda self, n: NAMES.index(n)
        Engine.get_binding_shape = lambda self, i: in_shape() if i == 0 else (in_shape()[0], 300, 6)
        Engine.get_binding_dtype = lambda self, i: np.float32
        Engine.get_profile_shape = lambda self, p, i: ((1, 3, 32, 32), (4, 3, T, T), (MAXB, 3, 2 * T, 2 * T))

    class Runtime:
        def __init__(self, logger): pass
        def deserialize_cuda_engine(self, blob):
            return Engine() if blob.startswith(b"FAKEENGINE") else None
    trt.Runtime = Runtime
    return trt, MAXB


def load_07():
    spec = importlib.util.spec_from_file_location("bench07", ROOT / "07_edge_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cpu_backend(b07):
    class CPUTRT(b07.TensorRTInferencer):
        def _check_cuda(self): pass
        def _device(self): return self.torch.device("cpu")
        def _make_stream(self): return types.SimpleNamespace(cuda_stream=0, synchronize=lambda: None)
        def _stream_ctx(self): return contextlib.nullcontext()
    return CPUTRT


def write_engine(d: Path, header: bool) -> Path:
    p = d / ("ultra.engine" if header else "bare.engine")
    body = b"FAKEENGINE" + bytes(range(256))
    if header:
        meta = json.dumps({"imgsz": [T, T], "batch": 4, "names": {"0": "crt_monitor"}}).encode()
        p.write_bytes(len(meta).to_bytes(4, "little", signed=True) + meta + body)
    else:
        p.write_bytes(body)
    return p


def main():
    b07 = load_07()
    rng = np.random.default_rng(0)
    n_checks = 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for new_api in (True, False):
            for dynamic in (True, False):
                for header in (True, False):
                    trt, maxb = make_fake_trt(new_api, dynamic)
                    sys.modules["tensorrt"] = trt
                    inf = cpu_backend(b07)(write_engine(td, header))
                    assert inf.max_batch == maxb and inf.input_size == T and inf.dynamic == dynamic
                    assert bool(inf.metadata) == header
                    for b in sorted({1, 3, maxb}):
                        x = rng.random((b, 3, T, T), dtype=np.float32)
                        out = inf.infer(x)[0]
                        assert out.shape == (b, 300, 6), out.shape
                        assert np.allclose(out[:, 0, 4], x.reshape(b, -1).mean(1), atol=1e-6), "batch order / padding"
                        # and through the tiled path: parse each item exactly as FrameDetector does
                        for i in range(b):
                            d = b07.et.parse_raw_output(out[i:i + 1], T, 1e-6)
                            assert len(d) == 1 and abs(d[0, 4] - x[i].mean()) < 1e-6
                        n_checks += 1
                    try:
                        inf.infer(np.zeros((maxb + 1, 3, T, T), np.float32))
                        raise AssertionError("oversized batch accepted")
                    except ValueError:
                        pass
                    print(f"  ok  api={'tensor' if new_api else 'binding':7s} batch={'dynamic' if dynamic else 'static ':7s} "
                          f"header={header}")
        # FrameDetector end-to-end on the fake engine (static batch 4 -> padding inside a frame)
        trt, _ = make_fake_trt(True, False)
        sys.modules["tensorrt"] = trt
        inf = cpu_backend(b07)(write_engine(td, True))
        fd = b07.FrameDetector(inf, T, 8, True, None)
        frame = (rng.random((70, 90, 3)) * 255).astype(np.uint8)
        dets = fd(frame, 0.0)
        assert fd.batch == 4 and fd.last_tiles > 4 and len(dets) > 0
        print(f"  ok  FrameDetector: {fd.last_tiles} tiles in batches of {fd.batch}, {len(dets)} merged detections")
    print(f"\nTensorRT backend logic: {n_checks} batch checks passed (fake TensorRT; real hardware untested)")


if __name__ == "__main__":
    main()
