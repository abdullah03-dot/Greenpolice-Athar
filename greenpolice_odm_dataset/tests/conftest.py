"""
pytest configuration (added 2026-11 with the athar/ layer).

Markers (registered in pytest.ini):
  gpu        needs torch / ultralytics / a CUDA device or TensorRT
  needs_data  needs datasets, weights or rendered assets that are not in the repository
  needs_torch needs torch + ultralytics installed (runs on CPU, but the install is ~1 GB)
CI runs   pytest -m "not gpu and not needs_data and not needs_torch".

Existing test files are scripts (python tests/<name>.py) that pytest also collects.  The ones that
need heavy dependencies are marked here by file name instead of being edited; when their imports are
missing they are not collected at all (an ImportError at collection is not a test result).
"""
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

# file -> (marker, modules it cannot be collected without)
HEAVY = {
    "test_onnx_parity.py": ("gpu", ["ultralytics", "torch", "onnxruntime"]),
    "test_trt_backend.py": ("gpu", ["torch"]),
    "test_mosaic_scale.py": ("needs_data", []),     # needs a tiled dataset on disk
}

# single tests inside otherwise-light files that call ultralytics (checked 2026-11-05: these are the
# only failures of the existing suite in an env without torch/ultralytics)
NODE_MARKS = {
    "test_ablation.py::test_epoch_cap": "needs_torch",
    "test_bootstrap_zeroshot.py::test_zeroshot": "needs_torch",
    "test_synth_splits.py::test_sources_and_guard": "needs_torch",
}


def _missing(mods):
    return [m for m in mods if importlib.util.find_spec(m) is None]


def pytest_ignore_collect(collection_path, config):
    spec = HEAVY.get(Path(str(collection_path)).name)
    if spec and _missing(spec[1]):
        return True
    return None          # never False: False would override --ignore / norecursedirs


def pytest_collection_modifyitems(config, items):
    for it in items:
        spec = HEAVY.get(Path(str(it.fspath)).name)
        if spec:
            it.add_marker(getattr(pytest.mark, spec[0]))
        m = NODE_MARKS.get(f"{Path(str(it.fspath)).name}::{it.name}")
        if m:
            it.add_marker(getattr(pytest.mark, m))


@pytest.fixture(scope="session")
def run11_template(tmp_path_factory):
    """A REAL 11_evidence_pipeline.py run (stills, 6 synthetic frames, the pixel-based fake detector of
    tests/test_fusion.py, no weights, CPU).  Returns (run_dir, keys) where the evidence head is signed
    with an ephemeral Ed25519 key kept in a temp dir."""
    import test_fusion as TF
    import sign_chain_head as SCH
    td = tmp_path_factory.mktemp("run11")
    TF.make_flight(td / "seq", n_frames=6)
    for n in ("a.pt", "b.pt"):
        (td / n).write_bytes(n.encode())
    k = SCH.keygen(td / "keys" / "evidence.pem")
    rk = SCH.keygen(td / "keys" / "reviewer.pem")
    out = td / "ev"
    ev = TF._load_11()
    argv = sys.argv
    sys.argv = ["11", "--weights", str(td / "a.pt"), str(td / "b.pt"), "--images", str(td / "seq" / "images"),
                "--tta", "none", "--out_dir", str(out), "--tile", "512", "--report_conf", "0.2",
                "--signing_pubkey", k["public_key"]]
    try:
        ev.main()
    finally:
        sys.argv = argv
    rd = out / (out / "LATEST_RUN.txt").read_text().strip()
    SCH.sign_head(rd / "evidence_audit.jsonl", Path(k["private_key"]))
    return rd, {"evidence": k, "reviewer": rk, "media": td / "seq" / "images"}


@pytest.fixture
def run11(run11_template, tmp_path):
    """A private copy of the run folder per test (tests tamper with it)."""
    src, keys = run11_template
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    return dst, keys
