# Model serving

The recon gateway calls four kinds of model. Each one runs as its own HTTP server, and the LiteLLM gateway
(`../gateway/`) puts them all behind one endpoint and one key.

| Model | What it does in a brief | Server | GPU used in our tests |
|---|---|---|---|
| Fire classifier | DINOv2 + LoRA fire probability, CLIP zero-shot second opinion, thermal tools on Celsius TIFFs | `fire_classifier/model.py` | T4, L4 or RTX PRO 6000 |
| Vision LLM | Plans extra image checks, answers yes/no image probes from log-probs, writes the brief | vLLM (flags below) | one RTX PRO 6000 per Qwen3.8 build, L4 for Qwen3-VL-4B |
| Kai 0.6B | Optional decision model: fire status from the evidence written as sentences | `kai/model.py` | L4 or RTX PRO 6000; small enough for a laptop |
| Clef 27B / Clef-Flash 9B | Optional decision models: same question, also see the first and last frame | `clef/model.py` | one RTX PRO 6000 each |
| d3 26B / d3-flash 8.4B / d3-mini 4.5B | Decision models that read the images directly; d3 is the first stage of the cascade (`experiments/decision-cascade/`) | `d3/model.py` | d3: one 80 GB+ card (H100, RTX PRO 6000); d3-flash and d3-mini: one L4 |

## Classifier, Kai, Clef and d3

Each `model.py` has a `Model` class with `load()` and `predict(dict) -> dict`. `http_server.py` wraps one in a small
FastAPI app with `POST /predict` and `GET /health`:

```bash
pip install fastapi uvicorn torch transformers peft
DETECTOR_DIR=path/to/weights python serving/http_server.py fire_classifier --port 8010
python serving/http_server.py kai --port 8011
CLEF_PATH=path/to/clef python serving/http_server.py clef --port 8012
CLEF_PATH=path/to/clef-flash python serving/http_server.py clef --port 8013
D3_PATH=path/to/d3 python serving/http_server.py d3 --port 8014
```

- **Fire classifier.** `DETECTOR_DIR` holds `detector/` and, optionally, `detector_giant/`. Each folder is a
  `final_<variant>/` folder written by `training/flame3-lora/train_cv.py` (adapter, `head.pt`, `serving_config.json`).
  Weights are not committed. Request: `{"frames": [{"id", "rgb_b64", "tiff_b64"?}], "models": [...]}`.
  `QUANT_VARIANTS="dinov2g_qlora:nvfp4"` adds a compiled 4-bit copy (Blackwell GPUs only; needs torchao).
- **Kai.** Loads `vllm-sr/Decision-2.0-Kai-0.6B` at a pinned revision through transformers. The newer revision fails
  its own file-hash check.
- **Clef.** `CLEF_PATH` is a local copy of the model repository (`Cloudflare/clef` or the Flash variant). The server
  calls the repository's own `joint_schema_model` code, so it needs that repository's requirements.
- **d3.** `D3_PATH` is a local copy of the model repository ([`vllm-sr/d3`](https://huggingface.co/vllm-sr/d3),
  [`d3-flash`](https://huggingface.co/vllm-sr/d3-flash) or [`d3-mini`](https://huggingface.co/vllm-sr/d3-mini), pinned to
  the revision you tested; the cascade trials list theirs). The server loads the repository's own `d3_runtime` and needs `torch`, `transformers==5.17.0`,
  `pillow`, `einops` and, for speed, `flash-linear-attention==0.5.2`. Images go through the runtime's strict checks
  (PNG, JPEG or WebP, at most 8 MB and 16 MP each, at most `D3_MAX_IMAGES`), video is refused, and a bad request gets
  422. `D3_BATCH_SIZE=4` keeps start-up memory low on a 24 GB card. The repository runs its own code at load, so
  read it before running a new revision.

Hosted deployment configs for these servers and the vLLM builds are in [`../deploy/`](../deploy/).

## Vision LLMs (vLLM 0.31.0)

All Qwen3.8 builds are served under the same name, `Qwen/Qwen3.8-27B`, so the gateway config only changes the URL.
Revisions are the ones we tested.

| Route | Weights (revision) | Extra flags |
|---|---|---|
| `qwen38-27b-bf16` | `Qwen/Qwen3.8-27B` (`1d4bf0f2`) | `--max-num-seqs 256` (the default 1024 needs more Mamba cache blocks than fit) |
| `qwen38-27b-fp8` | `Qwen/Qwen3.8-27B-FP8` (`017b9c7a`) | |
| `qwen38-27b-int4` | `RedHatAI/Qwen3.8-27B-INT4` (`91bd022d`) | |
| `qwen38-27b-nvfp4` | `Inferact/Qwen3.8-27B-NVFP4` (`6128240e`) | needs a Blackwell GPU |
| `qwen3vl-4b` | `Qwen/Qwen3-VL-4B-Instruct` (`ebb281ec`) | `--max-model-len 16384 --tool-call-parser hermes`, no reasoning parser |

Common flags for the Qwen3.8 builds (BF16 shown):

```bash
vllm serve Qwen/Qwen3.8-27B --served-model-name Qwen/Qwen3.8-27B --port 8001 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image": 6}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --max-logprobs 20 --enable-prefix-caching --max-num-seqs 256
```

Why these flags matter to the gateway:

- `--max-logprobs 20`: image probes score answer letters from the top 20 log-probs.
- `--enable-auto-tool-choice --tool-call-parser qwen3_coder`: the planner uses forced tool calls.
- `--reasoning-parser qwen3`: keeps any thinking text out of `content`. The gateway also turns thinking off per
  request with `chat_template_kwargs.enable_thinking=false`.
- Scored runs send one request at a time per server, because vLLM log-probs can shift with batch composition.
