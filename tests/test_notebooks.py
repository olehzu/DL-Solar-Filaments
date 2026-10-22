"""Smoke test: run both notebooks headlessly on CPU against synthetic data (no Kaggle, no downloads).

    pip install -r requirements.txt -r requirements-dev.txt
    pytest tests/ -v

Notebooks are executed from a temporary copy, so nothing (outputs, results JSON) is written into the repo.
"""
import os
import shutil
import sys

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

sys.path.insert(0, os.path.dirname(__file__))
from make_synthetic_coco import make  # noqa: E402

NB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    root = tmp_path_factory.mktemp("nbrun")
    data_dir, _, _ = make(str(root / "data"))
    mp = pytest.MonkeyPatch()
    for k, v in {"DATA_DIR": data_dir, "OUTPUT_DIR": str(root / "outputs"), "EPOCHS": "1", "BATCH": "2",
                 "ENCODER_WEIGHTS": "None", "MPLBACKEND": "Agg"}.items():
        mp.setenv(k, v)
    yield root
    mp.undo()


def _run(name, workdir, patch=None):
    nb = nbformat.read(os.path.join(NB_DIR, name), as_version=4)
    if patch:
        for c in nb.cells:
            if c.cell_type == "code":
                for old, new in patch.items():
                    c.source = c.source.replace(old, new)
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(workdir)}})
    return nb


def _stdout(nb):
    return "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.outputs
                   if o.get("output_type") == "stream")


def _check_stats(out):
    line = next(l for l in out.splitlines() if l.startswith("annotations total over splits"))
    total, in_json = (int(s.split("=")[1].strip(" )")) for s in line.split("  ("))
    assert total == in_json, line
    assert "file names used by more than one image record: 2 (2 extra records)" in out


def test_01_logreg_baseline(workdir):
    out = _stdout(_run("01_logreg_baseline.ipynb", workdir))
    _check_stats(out)
    assert "Leakage checks passed" in out and "LogReg  test:" in out and "Threshold test:" in out
    assert (workdir / "outputs" / "results_baselines.json").exists()


def test_02_unet(workdir):
    # DataLoader worker processes cannot import classes defined in a notebook under Windows "spawn";
    # Colab/Linux uses fork, so the tracked notebook keeps num_workers=2 and only this temp copy is patched.
    patch = {"num_workers=2": "num_workers=0"} if os.name == "nt" else None
    out = _stdout(_run("02_unet.ipynb", workdir, patch))
    _check_stats(out)
    assert "Leakage checks passed" in out and "U-Net test:" in out
