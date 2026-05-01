#!/usr/bin/env python3
"""Emit one speed-accuracy CSV row for a trained SAMWISE Echo run."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import opts
from datasets.transform_utils import VideoEvalDataset
from inference_echo import dataset_name_from_video_id, parse_skip_datasets
from models.samwise import build_samwise
from util.misc import on_load_checkpoint


INTERNAL_TEST_DATASETS = (
    "Camus",
    "CardiacUDA",
    "EchoCP",
    "SegRWMA",
    "EchoNet-Pediatric",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[opts.get_args_parser()],
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("output/echo_refvos_train_test_epoch03/eval_echo/test/metrics_p2flow_per_video.json"),
        help="Local test metrics JSON produced for this run.",
    )
    parser.add_argument("--model-name", default="SAMWISE", help="CSV model name.")
    parser.add_argument("--bubble-size", type=int, default=1000)
    parser.add_argument("--color", default="#5B6BAA")
    parser.add_argument("--marker", default="o")
    parser.add_argument("--label-dx", type=float, default=4)
    parser.add_argument("--label-dy", type=float, default=0.08)
    parser.add_argument("--group", default="baseline")
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--timed-batches", type=int, default=200)
    parser.add_argument(
        "--include-datasets",
        default=",".join(INTERNAL_TEST_DATASETS),
        help="Comma-separated datasets to include for timing and balanced Dice.",
    )
    return parser.parse_args()


def load_test_mdice_percent(metrics_json: Path, include_datasets: set[str]) -> float:
    with open(metrics_json, "r") as f:
        metrics = json.load(f)

    per_dataset: dict[str, list[float]] = {dataset: [] for dataset in include_datasets}
    for video in metrics["videos"].values():
        dataset = video["dataset"]
        if dataset not in include_datasets:
            continue
        for record in video["records"]:
            dice = record.get("dice")
            if dice is not None and math.isfinite(float(dice)):
                per_dataset[dataset].append(float(dice))

    missing = [dataset for dataset, values in per_dataset.items() if not values]
    if missing:
        raise ValueError(f"Missing Dice records for datasets: {missing}")

    return float(np.mean([np.mean(values) for values in per_dataset.values()]) * 100.0)


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = build_samwise(args)
    model.to(device)

    if not args.resume:
        raise ValueError("--resume is required for benchmarking a trained run")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    checkpoint = on_load_checkpoint(model, checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected_keys = [
        key
        for key in unexpected_keys
        if not (key.endswith("total_params") or key.endswith("total_ops"))
    ]
    if missing_keys:
        print(f"Missing Keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected Keys: {unexpected_keys}")

    model.eval()
    return model


def iter_echo_expression_clips(args: argparse.Namespace, include_datasets: set[str]):
    root = Path(args.ytvos_path)
    split = "valid" if args.split == "valid_u" else args.split
    img_folder = root / split / "JPEGImages"
    meta_file = root / "meta_expressions" / split / "meta_expressions.json"
    with open(meta_file, "r") as f:
        videos = json.load(f)["videos"]

    skip_datasets = parse_skip_datasets(args.skip_datasets)
    video_ids = [
        video_id
        for video_id in sorted(videos)
        if dataset_name_from_video_id(video_id) in include_datasets
        and dataset_name_from_video_id(video_id) not in skip_datasets
    ]

    for video_id in video_ids:
        video = videos[video_id]
        frames = video["frames"]
        vd = VideoEvalDataset(str(img_folder / video_id), frames, max_size=args.max_size)
        dl = DataLoader(
            vd,
            batch_size=args.eval_clip_window,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available() and str(args.device).startswith("cuda"),
        )
        for exp_id, exp_dict in sorted(video["expressions"].items(), key=lambda item: int(item[0])):
            expression = exp_dict["exp"]
            for imgs, clip_frame_ids in dl:
                frame_ids = [int(idx) for idx in clip_frame_ids.tolist()]
                frame_keys = [(video_id, frames[idx]) for idx in frame_ids]
                yield imgs, expression, frame_ids, frame_keys, str(exp_id)


def time_model(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> OrderedDict[str, Any]:
    use_cuda_events = device.type == "cuda"
    seen_timed_frames: set[tuple[str, str]] = set()
    timed_batches = 0
    timed_clips = 0
    total_model_seconds = 0.0

    include_datasets = {
        item.strip()
        for item in args.include_datasets.split(",")
        if item.strip()
    }
    clip_iter = iter_echo_expression_clips(args, include_datasets)

    precision = "fp32"
    timing_protocol = "CUDA-event timing around model forward only" if use_cuda_events else "perf_counter timing around model forward only"

    warmup_progress = tqdm(
        total=args.warmup_batches,
        desc="Warmup forwards",
        unit="batch",
        leave=False,
    )
    timed_progress = tqdm(
        total=args.timed_batches,
        desc="Timed forwards",
        unit="batch",
    )

    with torch.inference_mode():
        try:
            for batch_index, (imgs_cpu, expression, frame_ids, frame_keys, _exp_id) in enumerate(clip_iter):
                imgs = imgs_cpu.to(device, non_blocking=True)
                img_h, img_w = imgs.shape[-2:]
                target = {
                    "size": torch.as_tensor([int(img_h), int(img_w)], device=device),
                    "frame_ids": frame_ids,
                }
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

                if batch_index < args.warmup_batches:
                    _ = model([imgs], [expression], [target])
                    warmup_progress.update(1)
                    continue

                if timed_batches >= args.timed_batches:
                    break

                if use_cuda_events:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    _ = model([imgs], [expression], [target])
                    end_event.record()
                    torch.cuda.synchronize(device)
                    elapsed_seconds = start_event.elapsed_time(end_event) / 1000.0
                else:
                    start = time.perf_counter()
                    _ = model([imgs], [expression], [target])
                    elapsed_seconds = time.perf_counter() - start

                total_model_seconds += elapsed_seconds
                timed_batches += 1
                timed_clips += 1
                seen_timed_frames.update(frame_keys)
                timed_valid_frames = len(seen_timed_frames)
                timed_progress.update(1)
                timed_progress.set_postfix(
                    frames=timed_valid_frames,
                    model_seconds=f"{total_model_seconds:.2f}",
                    fps=f"{timed_valid_frames / total_model_seconds:.2f}" if total_model_seconds > 0 else "0.00",
                )
        finally:
            warmup_progress.close()
            timed_progress.close()

    timed_valid_frames = len(seen_timed_frames)
    fps = timed_valid_frames / total_model_seconds if total_model_seconds > 0 else 0.0
    return OrderedDict(
        model=args.model_name,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        batch_size=1,
        image_size=args.max_size,
        sequence_length=args.eval_clip_window,
        split=args.split,
        timed_batches=timed_batches,
        timed_clips=timed_clips,
        timed_valid_frames=timed_valid_frames,
        total_model_seconds=total_model_seconds,
        fps=fps,
        precision=precision,
        timing_protocol=timing_protocol,
    )


def csv_escape(value: Any) -> str:
    text = str(value)
    if any(char in text for char in [",", '"', "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def main() -> None:
    args = parse_args()
    args.batch_size = 1
    args.no_distributed = True
    args.distributed = False

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    include_datasets = {
        item.strip()
        for item in args.include_datasets.split(",")
        if item.strip()
    }

    dice_percent = load_test_mdice_percent(args.metrics_json, include_datasets)
    model = load_model(args, device)
    timing = time_model(args, model, device)
    timing["dice_percent"] = dice_percent
    timing["metric_json"] = str(args.metrics_json.resolve())

    notes = (
        f"GPU={timing['gpu']}; batch=1; image={args.max_size}; T={args.eval_clip_window}; "
        f"split={args.split}; timed_frames={timing['timed_valid_frames']}; "
        f"seconds={timing['total_model_seconds']:.4f}; precision={timing['precision']}; "
        f"metric_json={Path(args.metrics_json).resolve()}"
    )

    header = "model,fps,dice,bubble_size,color,marker,label_dx,label_dy,group,notes"
    row = [
        args.model_name,
        f"{timing['fps']:.2f}",
        f"{dice_percent:.2f}",
        args.bubble_size,
        args.color,
        args.marker,
        args.label_dx,
        args.label_dy,
        args.group,
        notes,
    ]

    print(header)
    print(",".join(csv_escape(value) for value in row))
    print()
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
