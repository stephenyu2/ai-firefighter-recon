# AI Firefighter: Wildfire Reconnaissance System

Mission software that turns a UAV's wildfire feed into a lightweight, structured reconnaissance brief for a CAL FIRE air attack supervisor. Instead of raw video, the system reports fire and smoke presence and extent, a relative description of how the fire is spreading and trending, and a confidence flag.

CIS 5980 AI Capstone (Fall 2026) - AI Engineering track

## Problem

During an active wildfire, an air attack supervisor reconstructs the fire's state from choppy radio traffic and raw drone video, a slow and error-prone process. Deployed wildfire AI today focuses on early detection ("is there a fire?") from fixed cameras and satellites. Little that is publicly deployed converts a drone's live aerial feed into the structured situational picture a supervisor needs once a fire is already located and crews are responding. That gap is what this project targets.

## What it does

- **Input:** UAV video and thermal imagery, processed frame-by-frame as if it were a live feed.
- **Output:** a structured reconnaissance brief containing
  - fire and smoke presence and extent (from a fine-tuned detector),
  - relative spread direction and trend (from an open vision-language model, with a deterministic baseline for comparison),
  - a confidence flag.

The system is decision-support, not autonomous dispatch: a human stays in the loop for any action taken on the brief.

## Approach

- **Detection:** fine-tune a pretrained detector for fire/smoke, with the decision threshold tuned for recall. Backbone comparison across YOLO / ResNet / MobileNet on the accuracy-versus-size tradeoff.
- **Spread direction and trend:** prompt Molmo 2 (open-weight vision-language model) to describe fire movement across frames, evaluated against a deterministic mask-tracking baseline.
- Frame handling and the baseline use OpenCV.

## Data

- **Primary:** FLAME 3 (radiometric thermal UAV imagery with georeferenced fire-progression data).
- **Cross-validation:** Boreal Forest Fire dataset (UAV imagery from independent controlled burns), used to test generalization across terrain and camera conditions.
- No training set is hand-labeled by the team. Both datasets are public research imagery of vegetation fires with no personal data.

FLAME 3 requires citation per its dataset terms (see Acknowledgements). The Boreal Forest Fire dataset is open access.

## Evaluation

- **Detection:** recall (primary, given the cost of a missed fire), plus precision and F1 on held-out FLAME 3 and Boreal splits.
- **Spread direction/trend:** agreement with ground truth (accuracy and Cohen's kappa) versus the deterministic baseline.
- **Product goal:** the brief should be readable and actionable in well under the time it takes to review the raw clip.

## Recon gateway (Milestone 2)

The pipeline that produces the brief lives in `src/recon_gateway/`. For a sequence of frames, the gateway:

1. runs the fine-tuned fire detector, a CLIP second opinion and the thermal tools on every frame;
2. tracks the hot area for trend and direction;
3. has a vision LLM check a few frames and write the brief as schema-checked JSON;
4. checks the brief against the measurements, so unsupported claims become "indeterminate".

Every model call goes through one LiteLLM gateway. How it works: [`docs/gateway.md`](docs/gateway.md).
Example output: [`examples/briefs/`](examples/briefs/).

Results so far:

| Experiment | Question | Report |
|---|---|---|
| [`experiments/flame3-baselines`](experiments/flame3-baselines/) | Which detector and protocol for fire/no-fire on FLAME 3 | `report/flame3_results.html` |
| [`experiments/full-burn-benchmark`](experiments/full-burn-benchmark/) | How fast and how accurate is the detector over a whole burn, on which GPU and precision, and on a site it never saw | `report/full_burn_report.html` |
| [`experiments/gateway-model-trials`](experiments/gateway-model-trials/) | Which LLM build should run the gateway, should a decision model make the fire call, and does an observer guidance prompt help | `report/gateway_model_trials.html` |
| [`experiments/decision-cascade`](experiments/decision-cascade/) | Can the d3 decision models read the images and make the fire call, and does a cascade (d3 first, Qwen3.8 on low confidence) hold up over a whole burn | `report/decision_cascade_trials.html` |

Main findings so far:

- **Thermal:** with thermal data, every Qwen3.8 build gets fire status right on every thermal brief.
- **Small RGB-only fires** are the open problem. The detectors and every LLM's own call miss most of them. The
  measurement rule on guided BF16 image checks caught 4 of 7 with no false alarms, but it needs more no-fire RGB footage
  before it can override the writeup.
- **Speed:** a whole burn takes seconds of GPU time, and uploads dominate end-to-end time.
- **Decision cascade:** d3 deciding first and handing low-confidence windows to Qwen3.8 ties the best method on the test
  sequences (17 of 20) and called every labeled window of the Florida burn correctly, at 2.2 s per image. Its remaining
  misses are small RGB-only fires that d3 calls "no fire" with high confidence.

## Repository structure

```
.
├── .github/workflows/ci.yml   # continuous integration (pytest)
├── src/recon_gateway/         # the gateway: tools, agent loop, brief schema, policy checks, CLI
├── gateway/                   # LiteLLM proxy config, request logging, route checks
├── serving/                   # model servers (fire classifier, Kai, Clef) and vLLM settings
├── deploy/                    # hosted deployment and training-job configs (see deploy/README.md)
├── training/                  # detector fine-tuning (LoRA/QLoRA, purged CV) and the precision study
├── experiments/               # baselines, full-burn benchmark, gateway model trials (scripts, results, reports)
├── examples/briefs/           # sample briefs
├── docs/                      # design notes
├── webapp/                    # replay console demo
├── scripts/                   # helper scripts
├── tests/                     # smoke test and offline gateway tests
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── LICENSE
```

## Getting started

```bash
git clone https://github.com/stephenyu2/ai-firefighter-recon
cd ai-firefighter-recon
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Use Python 3.11 (what CI runs). The tests run offline. To make a brief you need:

- a fire classifier server and a vision LLM (see [`serving/`](serving/));
- the LiteLLM gateway in front of them (see [`gateway/`](gateway/)). The command below assumes it; the gateway also turns
  off Qwen3.8 thinking text on every request.

```bash
export RECON_CLASSIFIER_URL=http://127.0.0.1:4000/recon/classifier
export RECON_LLM_URL=http://127.0.0.1:4000/v1
export RECON_GATEWAY_KEY=your-gateway-key
PYTHONPATH=src python -m recon_gateway --frames path/to/rgb --tiffs path/to/tiffs --model qwen38-27b-fp8 --out brief
```

This writes `brief.md` and `brief.json`. To skip the gateway, `unset RECON_GATEWAY_KEY` (it takes precedence), point the URLs at
the servers, set `RECON_API_KEY` if they need one, and pass the served model name (`--model Qwen/Qwen3.8-27B`); thinking then stays at the server's default. Datasets go under `data/` (git-ignored) or wherever `RECON_DATA_ROOT`
points.

## Team and roles

| Member | Area |
| --- | --- |
| Caroline O'Sullivan | Data (FLAME 3 + Boreal acquisition, preprocessing, splits); Documentation / repository |
| Spencer Theissen-Vang | Modeling (backbone selection and comparison, fine-tuning, recall-tuned thresholding) |
| Stephen Yu | Modeling (deterministic mask-tracking baseline; Molmo 2 integration and prompting) |
| Joe Lucas | Evaluation (metrics, test set, trend scoring); Application (assembling the brief; demo) |

## License

Released under the MIT License. See [LICENSE](LICENSE). Dataset licenses are separate and governed by their respective providers.

## Acknowledgements

FLAME 3 dataset: Hopkins, B., O'Neill, L., Marinaccio, M., Rowell, E., Parsons, R., Flanary, S., Nazim, I., Seielstad, C., Afghah, F. (2024). *FLAME 3 Dataset: Unleashing the Power of Radiometric Thermal UAV Imagery for Wildfire Management.* arXiv:2412.02831.
