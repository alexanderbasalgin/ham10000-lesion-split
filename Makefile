PY ?= python

.PHONY: test sample quick train serve

test:
	$(PY) -m pytest -q

# ~300 images from the Hugging Face mirror, enough to smoke-test the pipeline
sample:
	$(PY) scripts/fetch_sample.py --out data/sample --n 300

quick: sample
	DATA_ROOT=data/sample OUTPUT_DIR=artifacts/quick QUICK=1 $(PY) kaggle/train.py

# full run, needs the Kaggle HAM10000 copy under DATA_ROOT and a GPU (~1.5 h on a T4)
train:
	DATA_ROOT=$${DATA_ROOT:-data/ham10000} OUTPUT_DIR=artifacts/run $(PY) kaggle/train.py

serve:
	uvicorn app.main:app --port 8000
