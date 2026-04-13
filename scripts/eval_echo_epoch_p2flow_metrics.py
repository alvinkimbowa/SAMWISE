#!/usr/bin/env python3
"""Score SAMWISE Echo validation outputs with P2Flow's metric implementation."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samwise-output-dir",
        type=Path,
        default=Path("output"),
        help="Experiment output directory containing valid_epochXX folders.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch index to score, used to resolve valid_epochXX under --samwise-output-dir.",
    )
    parser.add_argument(
        "--epoch-dir",
        type=Path,
        default=None,
        help="Explicit epoch directory to score, e.g. output/exp/valid_epoch00.",
    )
    parser.add_argument(
        "--ytvos-path",
        type=Path,
        default=Path("data/echo-ref-vos"),
        help="Echo Ref-VOS root used for GT masks and metadata.",
    )
    parser.add_argument(
        "--p2flow-root",
        type=Path,
        default=Path("/home/ultrai/UltrAi/moein/P2Flow"),
        help="Sibling P2Flow repo root to import metrics from.",
    )
    return parser.parse_args()


def resolve_epoch_dir(*, samwise_output_dir: Path, epoch: int | None, epoch_dir: Path | None) -> Path:
    if epoch_dir is not None and epoch is not None:
        raise ValueError("Provide only one of --epoch or --epoch-dir.")
    if epoch_dir is not None:
        return epoch_dir.resolve()
    if epoch is None:
        raise ValueError("Provide either --epoch or --epoch-dir.")
    return (samwise_output_dir / f"valid_epoch{epoch:02d}").resolve()


def load_module(module_name: str, module_path: Path):
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find module at {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_video_id(video_id: str) -> tuple[str, str, str]:
    parts = video_id.split("__", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Expected Echo video_id in 'dataset__view__clip' format, got {video_id!r}"
        )
    dataset, view, clip_id = parts
    return dataset, view, clip_id


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def json_float(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def make_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("dice", "iou", "hd95")
    summary: dict[str, Any] = {"n_rows": len(rows)}
    for metric_name in metrics:
        values = [float(row[metric_name]) for row in rows if math.isfinite(float(row[metric_name]))]
        summary[f"{metric_name}_n"] = len(values)
        summary[f"{metric_name}_mean"] = float(np.mean(values)) if values else None
        summary[f"{metric_name}_std"] = float(np.std(values)) if values else None
    return summary


def summarize_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {group_value: make_summary(group_rows) for group_value, group_rows in sorted(grouped.items())}


def main() -> None:
    args = parse_args()

    epoch_dir = resolve_epoch_dir(
        samwise_output_dir=args.samwise_output_dir,
        epoch=args.epoch,
        epoch_dir=args.epoch_dir,
    )
    pred_root = epoch_dir / "eval_echo" / "valid"
    gt_root = args.ytvos_path / "valid" / "Annotations"
    meta_path = args.ytvos_path / "meta_expressions" / "valid" / "meta_expressions.json"

    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_root}")
    if not gt_root.exists():
        raise FileNotFoundError(f"GT annotation directory not found: {gt_root}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Echo metadata not found: {meta_path}")

    p2flow_metrics = load_module(
        "p2flow_eval_metrics",
        args.p2flow_root / "src" / "metrics.py",
    )

    compute_case_metrics = p2flow_metrics.compute_case_metrics
    id_to_label = p2flow_metrics.ID_TO_LABEL

    metadata = read_json(meta_path)["videos"]

    csv_rows: list[dict[str, Any]] = []
    per_video_records: dict[str, dict[str, Any]] = {}
    scored_videos = 0
    skipped_missing_prediction_dirs = 0
    skipped_missing_prediction_frames = 0

    video_ids = sorted(metadata.keys())
    progress = tqdm(video_ids, desc="Scoring videos", ncols=0)

    for video_id in progress:
        video_pred_dir = pred_root / video_id
        if not video_pred_dir.exists():
            progress.set_postfix(
                scored_videos=scored_videos,
                rows=len(csv_rows),
                missing_dirs=skipped_missing_prediction_dirs,
                missing_frames=skipped_missing_prediction_frames,
            )
            continue

        dataset, view, clip_id = parse_video_id(video_id)
        expressions = metadata[video_id]["expressions"]
        frame_records: list[dict[str, Any]] = []

        for exp_id, exp_meta in sorted(expressions.items(), key=lambda item: int(item[0])):
            exp_pred_dir = video_pred_dir / exp_id
            if not exp_pred_dir.exists():
                skipped_missing_prediction_dirs += 1
                continue

            obj_id = int(exp_meta["obj_id"])
            if obj_id == 0:
                continue
            class_name = str(id_to_label[obj_id])

            has_prediction = False
            for frame_name in metadata[video_id]["frames"]:
                gt_path = gt_root / video_id / f"{frame_name}.png"
                if not gt_path.exists():
                    raise FileNotFoundError(f"Missing GT mask for {video_id}/{frame_name}: {gt_path}")

                gt_mask = read_mask(gt_path)
                if not np.any(gt_mask == obj_id):
                    continue

                pred_path = exp_pred_dir / f"{frame_name}.png"
                if not pred_path.exists():
                    skipped_missing_prediction_frames += 1
                    continue

                has_prediction = True
                pred_mask = read_mask(pred_path)
                pred_binary = (pred_mask > 0).astype(np.uint8)
                pred_single_class = pred_binary * np.uint8(obj_id)

                metrics = compute_case_metrics(pred_single_class, gt_mask, obj_id)
                row = {
                    "video_id": video_id,
                    "dataset": dataset,
                    "view": view,
                    "clip_id": clip_id,
                    "exp_id": str(exp_id),
                    "obj_id": obj_id,
                    "class_name": class_name,
                    "frame": frame_name,
                    "dice": float(metrics["dice"]),
                    "iou": float(metrics["iou"]),
                    "hd95": float(metrics["hd95"]),
                }
                csv_rows.append(row)
                frame_records.append(row)

            if not has_prediction:
                skipped_missing_prediction_dirs += 1

        if frame_records:
            scored_videos += 1
            per_video_records[video_id] = {
                "dataset": dataset,
                "view": view,
                "clip_id": clip_id,
                "records": [
                    {
                        **record,
                        "dice": json_float(record["dice"]),
                        "iou": json_float(record["iou"]),
                        "hd95": json_float(record["hd95"]),
                    }
                    for record in frame_records
                ],
                "summary_by_expression": summarize_by_key(frame_records, "exp_id"),
                "summary_by_class": summarize_by_key(frame_records, "class_name"),
            }

        progress.set_postfix(
            scored_videos=scored_videos,
            rows=len(csv_rows),
            missing_dirs=skipped_missing_prediction_dirs,
            missing_frames=skipped_missing_prediction_frames,
        )

    csv_path = pred_root / "metrics_p2flow_per_video.csv"
    json_path = pred_root / "metrics_p2flow_per_video.json"

    fieldnames = [
        "video_id",
        "dataset",
        "view",
        "clip_id",
        "exp_id",
        "obj_id",
        "class_name",
        "frame",
        "dice",
        "iou",
        "hd95",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    payload = {
        "epoch_dir": str(epoch_dir),
        "prediction_root": str(pred_root),
        "gt_root": str(gt_root),
        "meta_path": str(meta_path),
        "n_videos_scored": scored_videos,
        "n_expression_frames_scored": len(csv_rows),
        "n_missing_prediction_dirs_or_empty": skipped_missing_prediction_dirs,
        "n_missing_prediction_frames": skipped_missing_prediction_frames,
        "videos": per_video_records,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Scored videos: {scored_videos}")
    print(f"Scored expression-frames: {len(csv_rows)}")
    print(f"Skipped missing/empty expression dirs: {skipped_missing_prediction_dirs}")
    print(f"Skipped missing prediction frames: {skipped_missing_prediction_frames}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
