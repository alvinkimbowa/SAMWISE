#!/usr/bin/env python3
"""Convert the echo benchmark into the SAMWISE Ref-VOS directory layout."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


VALID_LABEL_IDS = {0, 1, 2, 3, 4, 5}


def parse_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/Dataset-Echocardiography-MPS-Video"),
        help="Root directory of the preprocessed echo benchmark.",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path("data/Dataset-Echocardiography-MPS-Video/data_splits.json"),
        help="Path to the benchmark split JSON used by P2Flow.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/echo-ref-vos"),
        help="Output directory in SAMWISE Ref-VOS layout.",
    )
    parser.add_argument(
        "--p2flow-root",
        type=Path,
        default=Path("/home/ultrai/UltrAi/moein/P2Flow"),
        help="Root of the local P2Flow repo used for prompt generation.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality for converted RGB frames.",
    )
    parser.add_argument(
        "--overwrite",
        type=parse_bool,
        default=False,
        help="Whether to overwrite an existing output root.",
    )
    return parser.parse_args()


def load_prompt_library(p2flow_root: Path):
    prompt_library_path = p2flow_root / "src" / "prompting" / "prompt_library.py"
    if not prompt_library_path.exists():
        raise FileNotFoundError(f"Could not find prompt library at {prompt_library_path}")

    spec = importlib.util.spec_from_file_location("p2flow_prompt_library", prompt_library_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {prompt_library_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_input_path(source_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    raw = str(raw_path)
    cleaned = raw[2:] if raw.startswith("./") else raw
    candidate = Path(cleaned)
    if candidate.exists():
        return candidate.resolve()

    prefixed = source_root.parent / cleaned
    if prefixed.exists():
        return prefixed.resolve()

    nested = source_root / cleaned
    if nested.exists():
        return nested.resolve()

    raise FileNotFoundError(f"Could not resolve input path {raw_path!r} relative to {source_root}")


def sanitize_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return sanitized.strip("._") or "unknown"


def build_video_id(dataset: str, plane: str, clip_id: str) -> str:
    return "__".join(
        [sanitize_component(dataset), sanitize_component(plane), sanitize_component(clip_id)]
    )


def sort_frame_stems(stems: list[str]) -> list[str]:
    def key(stem: str) -> tuple[int, str]:
        return (0, f"{int(stem):08d}") if stem.isdigit() else (1, stem)

    return sorted(stems, key=key)


def list_frame_map(directory: Path) -> dict[str, Path]:
    frames = {}
    for path in directory.iterdir():
        if path.is_file():
            frames[path.stem] = path
    return frames


def load_clip_masks(mask_dir: Path, frame_stems: list[str]) -> tuple[dict[str, np.ndarray], dict[int, list[str]], set[int]]:
    mask_arrays: dict[str, np.ndarray] = {}
    class_frames: dict[int, list[str]] = {label_id: [] for label_id in range(1, 6)}
    present_labels: set[int] = set()

    for stem in frame_stems:
        mask_array = np.array(Image.open(mask_dir / f"{stem}.png"))
        unique_values = set(int(x) for x in np.unique(mask_array).tolist())
        invalid_values = sorted(unique_values - VALID_LABEL_IDS)
        if invalid_values:
            raise ValueError(f"Mask {mask_dir / f'{stem}.png'} contains unsupported labels {invalid_values}")

        mask_arrays[stem] = mask_array.astype(np.uint8, copy=False)
        for label_id in range(1, 6):
            if np.any(mask_array == label_id):
                class_frames[label_id].append(stem)
                present_labels.add(label_id)

    return mask_arrays, class_frames, present_labels


def write_rgb_frames(
    *,
    image_map: dict[str, Path],
    frame_stems: list[str],
    out_dir: Path,
    jpg_quality: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in frame_stems:
        rgb = Image.open(image_map[stem]).convert("RGB")
        rgb.save(out_dir / f"{stem}.jpg", format="JPEG", quality=jpg_quality)


def write_masks(*, mask_arrays: dict[str, np.ndarray], frame_stems: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in frame_stems:
        mask = Image.fromarray(mask_arrays[stem].astype(np.uint8, copy=False))
        mask.save(out_dir / f"{stem}.png")


def ensure_empty_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_root} already exists. Re-run with --overwrite true to rebuild it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def split_output_name(split_name: str) -> str:
    return {"train": "train", "val": "valid", "test": "test"}[split_name]


def make_expression_payload(*, prompt_library: Any, plane: str, dataset: str, label_id: int) -> str:
    spec = prompt_library.PromptSpec(label_ids=(label_id,))
    return prompt_library.build_prompt_text(
        plane=plane,
        dataset=dataset,
        spec=spec,
        prompt_mode="view_structure_anatomy",
    )


def build_split_records(
    *,
    split_items: list[dict[str, Any]],
    split_name: str,
    source_root: Path,
    output_root: Path,
    prompt_library: Any,
    jpg_quality: int,
    report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    videos_meta: dict[str, Any] = {}
    videos_expr: dict[str, Any] = {}
    output_split = split_output_name(split_name)
    split_dir = output_root / output_split
    image_root = split_dir / "JPEGImages"
    mask_root = split_dir / "Annotations"

    seen_video_ids: set[str] = set()

    for item in tqdm(split_items, desc=f"convert {split_name}", ncols=0):
        dataset = str(item["dataset"])
        plane = str(item["plane"]).upper()
        clip_id = str(item["clip_id"])
        video_id = build_video_id(dataset, plane, clip_id)
        if video_id in seen_video_ids:
            raise ValueError(f"Duplicate video_id generated for split {split_name}: {video_id}")
        seen_video_ids.add(video_id)

        images_dir = resolve_input_path(source_root, item["images_dir"])
        masks_dir = resolve_input_path(source_root, item["masks_dir"])
        metadata_path = resolve_input_path(source_root, item["metadata_path"])
        metadata = json.loads(metadata_path.read_text())

        image_map = list_frame_map(images_dir)
        mask_map = list_frame_map(masks_dir)
        shared_stems = sort_frame_stems(sorted(set(image_map) & set(mask_map)))
        if not shared_stems:
            report["skipped_clips"].append(
                {
                    "split": split_name,
                    "video_id": video_id,
                    "reason": "no_shared_frames",
                    "images_dir": str(images_dir),
                    "masks_dir": str(masks_dir),
                }
            )
            continue

        dropped_images = sorted(set(image_map) - set(shared_stems))
        dropped_masks = sorted(set(mask_map) - set(shared_stems))
        if dropped_images or dropped_masks:
            report["frame_mismatches"].append(
                {
                    "split": split_name,
                    "video_id": video_id,
                    "dropped_image_frames": dropped_images,
                    "dropped_mask_frames": dropped_masks,
                }
            )

        mask_arrays, class_frames, present_label_ids = load_clip_masks(masks_dir, shared_stems)
        dataset_present = set(prompt_library.present_label_set(dataset))
        view_visible = set(prompt_library.view_label_set(plane))
        actual_present = {prompt_library.ID_TO_LABEL[label_id] for label_id in present_label_ids}
        active_short_labels = dataset_present & view_visible & actual_present

        object_entries: dict[str, Any] = {}
        expression_entries: dict[str, Any] = {}
        for label_id in range(1, 6):
            short_name = prompt_library.ID_TO_LABEL[label_id]
            if short_name not in active_short_labels:
                continue
            frames_for_object = class_frames[label_id]
            if not frames_for_object:
                continue

            object_entries[str(label_id)] = {
                "category": prompt_library.ID_TO_LONG[label_id],
                "frames": frames_for_object,
            }
            expression_entries[str(len(expression_entries))] = {
                "exp": make_expression_payload(
                    prompt_library=prompt_library,
                    plane=plane,
                    dataset=dataset,
                    label_id=label_id,
                ),
                "obj_id": str(label_id),
            }
            report["per_class_expression_counts"][str(label_id)] += 1

        if not expression_entries:
            report["skipped_clips"].append(
                {
                    "split": split_name,
                    "video_id": video_id,
                    "reason": "no_active_labels",
                    "dataset": dataset,
                    "plane": plane,
                    "present_label_ids": sorted(present_label_ids),
                    "metadata_label_values_present": metadata.get("label_values_present", []),
                }
            )
            continue

        write_rgb_frames(
            image_map=image_map,
            frame_stems=shared_stems,
            out_dir=image_root / video_id,
            jpg_quality=jpg_quality,
        )
        write_masks(
            mask_arrays=mask_arrays,
            frame_stems=shared_stems,
            out_dir=mask_root / video_id,
        )

        videos_meta[video_id] = {"objects": object_entries}
        videos_expr[video_id] = {
            "frames": shared_stems,
            "expressions": expression_entries,
        }

        report["split_counts"][split_name]["exported_videos"] += 1
        report["split_counts"][split_name]["frames"] += len(shared_stems)
        report["split_counts"][split_name]["expressions"] += len(expression_entries)

    return {"videos": videos_meta}, {"videos": videos_expr}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_validation_checks(*, output_root: Path) -> None:
    for split_name, output_split in (("train", "train"), ("val", "valid"), ("test", "test")):
        meta_path = output_root / output_split / "meta.json"
        expr_split = "val" if split_name == "val" else split_name
        expr_path = output_root / "meta_expressions" / expr_split / "meta_expressions.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta.json for split {split_name}: {meta_path}")
        if not expr_path.exists():
            raise FileNotFoundError(f"Missing meta_expressions.json for split {split_name}: {expr_path}")

        meta = json.loads(meta_path.read_text())["videos"]
        expr = json.loads(expr_path.read_text())["videos"]
        if set(expr) != set(meta):
            raise ValueError(f"Video ids do not match between {meta_path} and {expr_path}")

        split_dir = output_root / output_split
        for video_id, expr_payload in expr.items():
            frames = expr_payload["frames"]
            objects = meta[video_id]["objects"]
            for frame in frames:
                image_path = split_dir / "JPEGImages" / video_id / f"{frame}.jpg"
                mask_path = split_dir / "Annotations" / video_id / f"{frame}.png"
                if not image_path.exists() or not mask_path.exists():
                    raise FileNotFoundError(f"Missing exported frame for {video_id}/{frame}")
            for exp_payload in expr_payload["expressions"].values():
                obj_id = exp_payload["obj_id"]
                if obj_id not in objects:
                    raise ValueError(f"Expression obj_id {obj_id} missing from objects for {video_id}")
            for obj_id, obj_payload in objects.items():
                for frame in obj_payload["frames"]:
                    mask = np.array(Image.open(split_dir / "Annotations" / video_id / f"{frame}.png"))
                    if not np.any(mask == int(obj_id)):
                        raise ValueError(f"Object {obj_id} for {video_id} claims frame {frame} without pixels")

    valid_expr = json.loads(
        (output_root / "meta_expressions" / "valid" / "meta_expressions.json").read_text()
    )
    val_expr = json.loads(
        (output_root / "meta_expressions" / "val" / "meta_expressions.json").read_text()
    )
    if valid_expr != val_expr:
        raise ValueError("Validation metadata mismatch between meta_expressions/valid and meta_expressions/val")


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    split_json_path = args.split_json.resolve()
    output_root = args.output_root.resolve()
    p2flow_root = args.p2flow_root.resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    if not split_json_path.exists():
        raise FileNotFoundError(f"Split JSON does not exist: {split_json_path}")

    prompt_library = load_prompt_library(p2flow_root)
    split_json = json.loads(split_json_path.read_text())

    ensure_empty_output(output_root, args.overwrite)

    report: dict[str, Any] = {
        "source_root": str(source_root),
        "split_json": str(split_json_path),
        "output_root": str(output_root),
        "prompt_mode": "view_structure_anatomy",
        "ignored_splits": ["external"],
        "split_counts": {
            "train": {"source_items": 0, "exported_videos": 0, "frames": 0, "expressions": 0},
            "val": {"source_items": 0, "exported_videos": 0, "frames": 0, "expressions": 0},
            "test": {"source_items": 0, "exported_videos": 0, "frames": 0, "expressions": 0},
        },
        "per_class_expression_counts": Counter(),
        "skipped_clips": [],
        "frame_mismatches": [],
    }

    split_payloads: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for split_name in ("train", "val", "test"):
        split_items = list(split_json.get(split_name, []))
        report["split_counts"][split_name]["source_items"] = len(split_items)
        split_payloads[split_name] = build_split_records(
            split_items=split_items,
            split_name=split_name,
            source_root=source_root,
            output_root=output_root,
            prompt_library=prompt_library,
            jpg_quality=args.jpg_quality,
            report=report,
        )

    for split_name, (meta_payload, expr_payload) in split_payloads.items():
        output_split = split_output_name(split_name)
        write_json(output_root / output_split / "meta.json", meta_payload)
        expr_dir = "val" if split_name == "val" else split_name
        write_json(output_root / "meta_expressions" / expr_dir / "meta_expressions.json", expr_payload)
        if split_name == "val":
            write_json(output_root / "meta_expressions" / "valid" / "meta_expressions.json", expr_payload)

    report["per_class_expression_counts"] = dict(report["per_class_expression_counts"])
    write_json(output_root / "conversion_report.json", report)

    run_validation_checks(output_root=output_root)
    print(f"Wrote Echo-VOS export to {output_root}")
    print(f"Conversion report: {output_root / 'conversion_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
