"""
Molmo 2 as the Tier 2 experimental comparison arm against the deterministic
mask-tracking baseline. Prompts the model with a sequence of raw RGB
frames (not segmentation masks - Molmo works on images directly) and asks
for a structured answer we can parse deterministically, rather than
free-form text we'd have to interpret by hand.

Same two-layer split as reference_baseline.py, for the same reason: this
sandbox can't download/run Molmo 2 itself (no Hugging Face network access
here), so the parsing logic - the part that actually needs to be correct
and testable - is isolated from the model-loading/inference layer you'll
run yourself with real GPU access.

  - build_prompt() / parse_molmo_response(): pure functions, fully
    testable with canned response strings, no model needed
  - Molmo2Client: thin wrapper around transformers; only exercised when
    you actually run this against the real model

LLM output is not guaranteed to match the requested format every time -
parse_molmo_response() is deliberately defensive (regex first, fuzzy
fallback, explicit "unparseable" rather than silently guessing) because a
silently wrong parse would corrupt your results table without any visible
failure.

Usage:
    pip install transformers torch accelerate
    python baselines/molmo2_baseline.py \
        --manifest data/tier2_eval_sequences_rgb.csv \
        --human_labels data/tier2_human_labels.csv \
        --model_path allenai/Molmo2-7B-0926 \
        --out_prefix results/predictions/tier2_molmo2

Expected --manifest columns: sequence_id, frame_index, frame_path
  (RGB frames, not masks - same sequences as mask_tracking_baseline.py
  ideally, so the two methods are compared on identical inputs)
"""
import argparse
import re
from pathlib import Path

import pandas as pd

DIRECTION_BINS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "no clear direction"]
TREND_LABELS = ["growing", "stable", "receding"]


def build_prompt(direction_bins=DIRECTION_BINS, trend_labels=TREND_LABELS) -> str:
    return (
        "You are shown a sequence of aerial drone frames of a wildfire, in time order. "
        "Based on how the fire's visible extent changes across these frames, answer in "
        "exactly this format and nothing else:\n\n"
        "Direction: <one of: " + ", ".join(direction_bins) + ">\n"
        "Trend: <one of: " + ", ".join(trend_labels) + ">\n\n"
        "\"Direction\" means which way the fire is expanding on screen, using compass bins "
        "where up/north is the top of the frame. Use \"no clear direction\" if the fire "
        "isn't clearly expanding toward one side. \"Trend\" means whether the fire's total "
        "visible extent is growing, holding steady, or shrinking across the sequence."
    )


def parse_molmo_response(text: str, direction_bins=DIRECTION_BINS, trend_labels=TREND_LABELS):
    """
    Returns {"direction": ..., "trend": ...}, each either a valid label or
    the string "unparseable" if the response couldn't be confidently read.
    Tries the exact requested format first, then falls back to a loose
    substring search so minor formatting drift (extra whitespace, a stray
    sentence) doesn't throw away an otherwise usable answer.
    """
    result = {"direction": "unparseable", "trend": "unparseable"}

    direction_match = re.search(r"direction\s*:\s*(.+)", text, re.IGNORECASE)
    trend_match = re.search(r"trend\s*:\s*(.+)", text, re.IGNORECASE)

    def match_label(raw_value: str, valid_labels):
        raw_value = raw_value.strip().splitlines()[0].strip().rstrip(".")

        # exact match first (case-insensitive)
        for label in valid_labels:
            if raw_value.lower() == label.lower():
                return label

        # fallback: label appears as a whole word/phrase in the line, using
        # word-boundary regex rather than plain substring search - a plain
        # "in" check lets single-letter bins like "N" false-match inside
        # ordinary words (e.g. "N" inside "based on the plume drift").
        # Longest labels first so "no clear direction" or "NW" win over a
        # shorter label that could also technically match.
        for label in sorted(valid_labels, key=len, reverse=True):
            pattern = r"\b" + re.escape(label) + r"\b"
            if re.search(pattern, raw_value, re.IGNORECASE):
                return label
        return "unparseable"

    if direction_match:
        result["direction"] = match_label(direction_match.group(1), direction_bins)
    else:
        # last-resort fallback: scan the whole response for any direction word
        result["direction"] = match_label(text, direction_bins)

    if trend_match:
        result["trend"] = match_label(trend_match.group(1), trend_labels)
    else:
        result["trend"] = match_label(text, trend_labels)

    return result


class Molmo2Client:
    """Thin wrapper around a Hugging Face Molmo 2 checkpoint. Lazily
    imports transformers/torch so the parsing logic above stays testable
    without either installed."""

    def __init__(self, model_path: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)
        self.device = device

    def describe_sequence(self, frame_paths, prompt: str) -> str:
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in frame_paths]
        inputs = self.processor.process(images=images, text=prompt)
        inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items()}
        output = self.model.generate_from_batch(inputs, max_new_tokens=100, do_sample=False)
        generated = output[0, inputs["input_ids"].size(1):]
        return self.processor.tokenizer.decode(generated, skip_special_tokens=True)


def run_molmo_baseline(manifest_df: pd.DataFrame, client: Molmo2Client, prompt: str):
    """
    manifest_df: columns sequence_id, frame_index, frame_path.
    Returns a DataFrame: sequence_id, trend_prediction, direction_prediction, raw_response.
    """
    rows = []
    for sequence_id, group in manifest_df.sort_values("frame_index").groupby("sequence_id"):
        frame_paths = group["frame_path"].tolist()
        response_text = client.describe_sequence(frame_paths, prompt)
        parsed = parse_molmo_response(response_text)
        rows.append({
            "sequence_id": sequence_id,
            "trend_prediction": parsed["trend"],
            "direction_prediction": parsed["direction"],
            "raw_response": response_text,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path,
                         help="CSV with sequence_id, frame_index, frame_path columns (RGB frames)")
    parser.add_argument("--human_labels", type=Path, default=None,
                         help="optional CSV with sequence_id, trend_label, direction_label")
    parser.add_argument("--model_path", default="allenai/Molmo2-7B-0926",
                         help="Hugging Face model id or local path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_prefix", required=True, type=Path)
    args = parser.parse_args()

    manifest_df = pd.read_csv(args.manifest)
    client = Molmo2Client(args.model_path, args.device)
    prompt = build_prompt()
    preds_df = run_molmo_baseline(manifest_df, client, prompt)

    raw_path = Path(f"{args.out_prefix}_raw_responses.csv")
    preds_df.to_csv(raw_path, index=False)  # always keep raw text for manual spot-checking

    unparseable = preds_df[(preds_df["trend_prediction"] == "unparseable") |
                            (preds_df["direction_prediction"] == "unparseable")]
    if len(unparseable) > 0:
        print(f"[warn] {len(unparseable)} of {len(preds_df)} sequences had an unparseable "
              f"response - see {raw_path} to inspect raw_response and adjust the prompt if this is common")

    if args.human_labels and args.human_labels.exists():
        labels_df = pd.read_csv(args.human_labels)
        merged = preds_df.merge(labels_df, on="sequence_id", how="inner")
        trend_out = merged[["sequence_id", "trend_label", "trend_prediction"]].rename(
            columns={"trend_label": "label", "trend_prediction": "prediction"})
        direction_out = merged[["sequence_id", "direction_label", "direction_prediction"]].rename(
            columns={"direction_label": "label", "direction_prediction": "prediction"})
    else:
        trend_out = preds_df[["sequence_id", "trend_prediction"]].rename(columns={"trend_prediction": "prediction"})
        direction_out = preds_df[["sequence_id", "direction_prediction"]].rename(columns={"direction_prediction": "prediction"})

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    trend_path = Path(f"{args.out_prefix}_trend.csv")
    direction_path = Path(f"{args.out_prefix}_direction.csv")
    trend_out.to_csv(trend_path, index=False)
    direction_out.to_csv(direction_path, index=False)
    print(f"wrote {trend_path} and {direction_path}")


if __name__ == "__main__":
    main()