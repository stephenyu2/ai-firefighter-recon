# LiteLLM gateway

Every model call in a brief goes through one local LiteLLM proxy. Client code then has one URL and one key,
upstream keys stay on the proxy, and every request is logged in one place with a run tag.

```
recon_gateway  ──>  LiteLLM proxy (127.0.0.1:4000)  ──>  vision LLM routes   /v1/chat/completions
                                                     ──>  pass-through routes /recon/classifier, /recon/kai,
                                                                              /recon/clef, /recon/clef-flash
```

## Setup

```bash
python -m venv .venv-litellm
.venv-litellm/bin/pip install "litellm[proxy]" pyyaml
cd gateway/litellm
cp config.example.yaml config.yaml
cp .env.example .env.local
chmod 600 .env.local
```

Then:

1. Edit `config.yaml` so each `api_base` and `target` points at your model servers (see `../serving/`).
2. Set `LITELLM_MASTER_KEY` in `.env.local`. Set `UPSTREAM_API_KEY` if your servers need a key; otherwise leave the
   placeholder `none` (an empty value makes LiteLLM's OpenAI client refuse to send the request).
3. Start the proxy with `PATH=../../.venv-litellm/bin:$PATH ./run_proxy.sh`.

`config.yaml` and `.env.local` are git-ignored.

Point the recon gateway at the proxy:

```bash
export RECON_GATEWAY_KEY=your-master-key
export RECON_LLM_URL=http://127.0.0.1:4000/v1
export RECON_CLASSIFIER_URL=http://127.0.0.1:4000/recon/classifier
export RECON_KAI_URL=http://127.0.0.1:4000/recon/kai
export RECON_CLEF_URL=http://127.0.0.1:4000/recon/clef
export RECON_CLEF_FLASH_URL=http://127.0.0.1:4000/recon/clef-flash
export RECON_D3_URL=http://127.0.0.1:4000/recon/d3
export RECON_D3_MINI_URL=http://127.0.0.1:4000/recon/d3-mini
```

Without the proxy, unset `RECON_GATEWAY_KEY` (it takes precedence), set `RECON_API_KEY` (and `RECON_AUTH_SCHEME`
if the server does not use `Bearer`) and point the URLs straight at the model servers. Leave the key unset for servers with no auth. Pass the served model name
(`--model Qwen/Qwen3.8-27B`), not a route name, and note that only the proxy turns Qwen3.8 thinking off.

## Practices built in

- **Fail loudly on parameters.** `drop_params: false` makes an unsupported parameter an error instead of a silent
  drop, so a model that ignores `logprobs` or `response_format` cannot quietly change results.
- **No thinking text.** The Qwen3.8 routes set `chat_template_kwargs.enable_thinking=false` on every request.
- **Fixed sampling.** The client sends temperature 0 and a fixed seed (`RECON_SEED`, default 1234) on every call.
- **Strict tool calls.** Planner calls use `tool_choice: "required"`. Frame ids are enums, unknown `decide` arguments
  are rejected (stray `finish` arguments are noted and ignored), and each tool error goes back to the model as a message instead of a silent fallback.
- **Strict output schema.** The writeup uses a strict JSON schema, and the schema has no free numeric fields.
- **Retries only where safe.** The gateway and decider clients retry 408, 429, 5xx and timeouts with the identical
  body. Any other 4xx fails at once.
- **Run tags.** Requests carry `x-recon-run` when a run tag is set (`RECON_RUN_TAG`, or `run_tag=`; the experiment
  scripts set one). `custom_callbacks.py` writes one line per request to
  `logs/requests.jsonl` with route, run tag, latency, tokens, `finish_reason` and tool-call names. That covers
  pass-through failures too.

## Route check

```bash
set -a; source .env.local; set +a
SMOKE_IMAGE=path/to/aerial.jpg python check_routes.py qwen38-27b-bf16 qwen38-27b-fp8
```

The script checks four things on each route:

1. a plain reply with no reasoning text;
2. image log-probs;
3. a tool call;
4. JSON-schema output.

A route that fails any of these will break the gateway.
