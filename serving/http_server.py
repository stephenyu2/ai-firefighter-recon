"""Minimal HTTP wrapper for the model servers in this folder: POST /predict with a JSON body calls Model.predict.

Usage: python serving/http_server.py fire_classifier --port 8010   (or kai, clef)
Needs fastapi and uvicorn. Each Model has load() and predict(dict) -> dict; any host that calls those works too."""
import argparse
import importlib.util
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI


def build(name):
    spec = importlib.util.spec_from_file_location(f"{name}_model", Path(__file__).parent / name / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.Model()
    model.load()
    app = FastAPI(title=f"recon {name}")

    @app.post("/predict")
    def predict(body: dict = Body(...)):
        return model.predict(body)

    @app.get("/health")
    def health():
        return {"ok": True}
    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name", choices=["fire_classifier", "kai", "clef"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8010)
    a = p.parse_args()
    uvicorn.run(build(a.name), host=a.host, port=a.port)
