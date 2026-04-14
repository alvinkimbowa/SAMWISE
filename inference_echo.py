"""
Inference/evaluation code for SAMWISE on Echo-VOS validation data.
"""

import argparse
import json
import os
import random
import sys
import time
from os.path import join
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import util.misc as utils
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
from datasets.transform_utils import VideoEvalDataset, vis_add_mask
from models.samwise import build_samwise
from tools.colormap import colormap
from tools.metrics import db_eval_boundary, db_eval_iou
from util.misc import on_load_checkpoint

color_list = colormap().astype("uint8").tolist()


def parse_skip_datasets(skip_datasets_arg):
    if not skip_datasets_arg:
        return set()
    return {
        item.strip()
        for item in skip_datasets_arg.split(",")
        if item.strip() and item.strip().lower() not in {"none", "null"}
    }


def dataset_name_from_video_id(video_id):
    return video_id.split("__", 1)[0]


def main(args):
    args.batch_size = 1
    print("Inference only supports for batch size = 1")
    print(args)

    seed = args.seed + utils.get_rank()
    utils.init_distributed_mode(args)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    args.log_file = join(args.output_dir, "log.txt")
    with open(args.log_file, "w") as fp:
        fp.writelines(" ".join(sys.argv) + "\n")
        fp.writelines(str(args.__dict__) + "\n\n")

    start_time = time.time()
    model = build_samwise(args)
    device = torch.device(args.device)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        checkpoint = on_load_checkpoint(model_without_ddp, checkpoint)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            checkpoint["model"], strict=False
        )
        unexpected_keys = [
            k
            for k in unexpected_keys
            if not (k.endswith("total_params") or k.endswith("total_ops"))
        ]
        if len(missing_keys) > 0:
            print("Missing Keys: {}".format(missing_keys))
        if len(unexpected_keys) > 0:
            print("Unexpected Keys: {}".format(unexpected_keys))

    result = eval_echo(args, model, output_dir)
    if utils.is_main_process():
        out_str = f"J&F: {result[2]}\tJ: {result[0]}\tF: {result[1]}"
        with open(args.log_file, "a") as fp:
            fp.writelines(out_str + "\n")
        print(out_str)

    print("Total inference time: %.4f s" % (time.time() - start_time))


def eval_echo(args, model, save_path_prefix):
    root = Path(args.ytvos_path)
    split = "valid" if args.split == "valid_u" else args.split
    img_folder = os.path.join(root, split, "JPEGImages")
    mask_folder = os.path.join(root, split, "Annotations")
    meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]

    skip_datasets = parse_skip_datasets(args.skip_datasets)
    video_list = [
        video_id
        for video_id in sorted(data.keys())
        if dataset_name_from_video_id(video_id) not in skip_datasets
    ]
    skipped_video_count = len(data) - len(video_list)
    if args.distributed:
        rank = utils.get_rank()
        world_size = utils.get_world_size()
        sub_video_list = video_list[rank::world_size]
    else:
        sub_video_list = video_list

    result_dir = os.path.join(save_path_prefix, "eval_echo", split)
    os.makedirs(result_dir, exist_ok=True)
    visualize_dir = os.path.join(save_path_prefix, f"{split}_images")
    if args.visualize:
        os.makedirs(visualize_dir, exist_ok=True)

    local_results = {}
    progress = tqdm(total=len(sub_video_list), ncols=0, disable=not utils.is_main_process())
    model.eval()

    for video in sub_video_list:
        expressions = data[video]["expressions"]
        frames = data[video]["frames"]
        video_results = {}
        for exp_id, exp_dict in expressions.items():
            exp = exp_dict["exp"]
            obj_id = int(exp_dict["obj_id"])
            save_path = os.path.join(result_dir, video, exp_id)
            expected_frame_paths = [os.path.join(save_path, frame_name + ".png") for frame_name in frames]
            if expected_frame_paths and all(os.path.exists(path) for path in expected_frame_paths):
                gt_masks = []
                for frame_name in frames:
                    gt_path = os.path.join(mask_folder, video, frame_name + ".png")
                    gt_mask = np.array(Image.open(gt_path).convert("P"))
                    gt_masks.append((gt_mask == obj_id).astype(np.uint8))
                gt_masks = np.stack(gt_masks, axis=0)

                all_pred_masks = []
                for pred_path in expected_frame_paths:
                    pred_mask = np.array(Image.open(pred_path).convert("L"))
                    all_pred_masks.append((pred_mask > 0).astype(np.uint8))
                all_pred_masks = np.stack(all_pred_masks, axis=0)

                j = float(db_eval_iou(gt_masks, all_pred_masks).mean())
                f = float(db_eval_boundary(gt_masks, all_pred_masks).mean())
                video_results[exp_id] = {"exp": exp, "obj_id": obj_id, "J": j, "F": f}
                continue

            all_pred_masks = []

            vd = VideoEvalDataset(join(img_folder, video), frames, max_size=args.max_size)
            dl = DataLoader(vd, batch_size=args.eval_clip_window, num_workers=args.num_workers, shuffle=False)
            origin_w, origin_h = vd.origin_w, vd.origin_h
            for imgs, clip_frames_ids in dl:
                clip_frames_ids = clip_frames_ids.tolist()
                imgs = imgs.to(args.device)
                img_h, img_w = imgs.shape[-2:]
                size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
                target = {"size": size, "frame_ids": clip_frames_ids}
                with torch.no_grad():
                    outputs = model([imgs], [exp], [target])
                pred_masks = outputs["pred_masks"].unsqueeze(0)
                pred_masks = F.interpolate(
                    pred_masks, size=(origin_h, origin_w), mode="bilinear", align_corners=False
                )
                pred_masks = (pred_masks.sigmoid() > args.threshold)[0].cpu()
                all_pred_masks.append(pred_masks)

            all_pred_masks = torch.cat(all_pred_masks, dim=0).numpy().astype(np.uint8)
            gt_masks = []
            for frame_name in frames:
                gt_path = os.path.join(mask_folder, video, frame_name + ".png")
                gt_mask = np.array(Image.open(gt_path).convert("P"))
                gt_masks.append((gt_mask == obj_id).astype(np.uint8))
            gt_masks = np.stack(gt_masks, axis=0)

            j = float(db_eval_iou(gt_masks, all_pred_masks).mean())
            f = float(db_eval_boundary(gt_masks, all_pred_masks).mean())
            video_results[exp_id] = {"exp": exp, "obj_id": obj_id, "J": j, "F": f}

            os.makedirs(save_path, exist_ok=True)
            for frame_name, pred_mask in zip(frames, all_pred_masks):
                mask = Image.fromarray(pred_mask.astype(np.float32) * 255).convert("L")
                mask.save(os.path.join(save_path, frame_name + ".png"))

            if args.visualize:
                save_visualize_path_dir = os.path.join(visualize_dir, video, str(exp_id))
                os.makedirs(save_visualize_path_dir, exist_ok=True)
                for frame_name, pred_mask in zip(frames, all_pred_masks):
                    source_img = Image.open(os.path.join(img_folder, video, frame_name + ".jpg")).convert("RGBA")
                    source_img = vis_add_mask(source_img, pred_mask, color_list[obj_id % len(color_list)])
                    source_img.save(os.path.join(save_visualize_path_dir, frame_name + ".png"))

        local_results[video] = video_results
        progress.update(1)

    gathered = utils.all_gather(local_results)
    merged = {}
    for result in gathered:
        merged.update(result)

    if utils.is_main_process():
        scores = [metrics for per_video in merged.values() for metrics in per_video.values()]
        j_score = float(np.mean([x["J"] for x in scores])) if scores else 0.0
        f_score = float(np.mean([x["F"] for x in scores])) if scores else 0.0
        jf_score = (j_score + f_score) / 2.0
        metrics_path = os.path.join(save_path_prefix, "eval_echo", split, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "J": j_score,
                    "F": f_score,
                    "J&F": jf_score,
                    "skip_datasets": sorted(skip_datasets),
                    "n_videos_after_dataset_filter": len(video_list),
                    "n_videos_skipped_by_dataset_filter": skipped_video_count,
                    "videos": merged,
                },
                f,
                indent=2,
            )
    else:
        j_score = f_score = jf_score = 0.0

    if args.distributed:
        metrics = torch.tensor([j_score, f_score, jf_score], device=args.device)
        torch.distributed.broadcast(metrics, src=0)
        j_score, f_score, jf_score = metrics.tolist()

    return [j_score, f_score, jf_score]


if __name__ == "__main__":
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser("SAMWISE Echo evaluation script", parents=[opts.get_args_parser()])
    args = parser.parse_args()
    name_exp = args.name_exp
    args.output_dir = os.path.join(args.output_dir, name_exp)
    main(args)
