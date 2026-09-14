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

## Repository structure

```
.
├── .github/workflows/ci.yml   # continuous integration smoke test
├── scripts/                   # helper scripts
├── src/                       # source code (added in Milestone 2)
├── tests/                     # tests, incl. the CI smoke test
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── LICENSE
```

## Getting started

> Note: the pipeline is under active development. This section will expand as the CLI tool takes shape.

```bash
# clone
git clone <repo-url>
cd ai-firefighter-recon

# (recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# install
pip install -r requirements.txt

# run the tests
pytest
```

## Team and roles

| Member | Area |
| --- | --- |
| Caroline O'Sullivan | Data (FLAME 3 + Boreal acquisition, preprocessing, splits); Documentation / repository |
| Spencer Thiessen | Modeling (backbone selection and comparison, fine-tuning, recall-tuned thresholding) |
| Stephen Yu | Modeling (deterministic mask-tracking baseline; Molmo 2 integration and prompting) |
| Joe Lucas | Evaluation (metrics, test set, trend scoring); Application (assembling the brief; demo) |

## License

Released under the MIT License. See [LICENSE](LICENSE). Dataset licenses are separate and governed by their respective providers.

## Acknowledgements

FLAME 3 dataset: Hopkins, B., O'Neill, L., Marinaccio, M., Rowell, E., Parsons, R., Flanary, S., Nazim, I., Seielstad, C., Afghah, F. (2024). *FLAME 3 Dataset: Unleashing the Power of Radiometric Thermal UAV Imagery for Wildfire Management.* arXiv:2412.02831.
