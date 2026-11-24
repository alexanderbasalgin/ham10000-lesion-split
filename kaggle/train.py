import os
import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skinlesion.experiments import run  # noqa: E402

warnings.filterwarnings("ignore", message="y_pred contains classes")

if __name__ == "__main__":
    data_root = os.environ.get("DATA_ROOT", "/kaggle/input")
    out_dir = os.environ.get("OUTPUT_DIR", "artifacts/run")
    quick = os.environ.get("QUICK") == "1"
    run(data_root, out_dir, quick=quick, log=lambda s: print(s, flush=True))
