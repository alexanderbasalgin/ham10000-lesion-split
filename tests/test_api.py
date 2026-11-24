import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client(tiny_ckpt, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tiny_ckpt))
    from app.main import app
    return TestClient(app)


def _jpeg():
    buf = io.BytesIO()
    Image.fromarray(np.full((60, 80, 3), 120, np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def test_valid_request(client):
    r = client.post("/predict", files={"image": ("x.jpg", _jpeg(), "image/jpeg")},
                    data={"age": "45", "sex": "male", "site": "back"})
    assert r.status_code == 200
    assert "prediction" in r.json()


def test_not_an_image(client):
    r = client.post("/predict", files={"image": ("x.jpg", b"hello", "image/jpeg")})
    assert r.status_code == 400
    assert "decode" in r.json()["detail"]


def test_bad_site(client):
    r = client.post("/predict", files={"image": ("x.jpg", _jpeg(), "image/jpeg")}, data={"site": "moon"})
    assert r.status_code == 422


def test_age_out_of_range(client):
    r = client.post("/predict", files={"image": ("x.jpg", _jpeg(), "image/jpeg")}, data={"age": "300"})
    assert r.status_code == 422
