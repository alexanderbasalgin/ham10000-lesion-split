import io
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from skinlesion.data import SITES
from skinlesion.predict import DEFAULT_CKPT, load_model, predict

app = FastAPI(title="skin lesion classifier")
MAX_BYTES = 10 * 1024 * 1024


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(
    image: UploadFile = File(...),
    age: float | None = Form(None, ge=0, le=120),
    sex: str | None = Form(None, pattern="^(male|female)$"),
    site: str | None = Form(None),
):
    if site is not None and site not in SITES:
        raise HTTPException(422, f"unknown site '{site}', expected one of {SITES}")
    data = await image.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 10 MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "could not decode image")
    model, size = load_model(os.environ.get("MODEL_PATH", DEFAULT_CKPT))
    return predict(img, age, sex, site, model, size)
