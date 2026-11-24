import os
import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skinlesion.experiments import run, run_repeats  # noqa: E402

warnings.filterwarnings("ignore", message="y_pred contains classes")

if __name__ == "__main__":
    data_root = os.environ.get("DATA_ROOT", "/kaggle/input")
    out_dir = os.environ.get("OUTPUT_DIR", "artifacts/run")
    quick = os.environ.get("QUICK") == "1"
    log = lambda s: print(s, flush=True)  # noqa: E731
    # STAGE=main: all models on split seed 0; STAGE=repeats: two more split seeds + 320px run
    if os.environ.get("STAGE", "repeats") == "main":
        run(data_root, out_dir, quick=quick, log=log)
    else:
        run_repeats(data_root, out_dir, quick=quick, log=log)
