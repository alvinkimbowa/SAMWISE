'''
Inference code for SAMWISE, on Ref-Youtube-VOS
Modified from DETR (https://github.com/facebookresearch/detr)
Ref-Davis17 does not support visualize
'''


import pandas as pd
from davis2017.evaluation import DAVISEvaluation
import sys
from models.samwise import build_samwise
from util.misc import on_load_checkpoint
import random
import time
from pathlib import Path
import numpy as np
import torch
from datasets.transform_utils import VideoEvalDataset
from torch.utils.data import DataLoader
from os.path import join
import util.misc as utils
import os
from PIL import Image
import torch.nn.functional as F
import json
import argparse
from tqdm import tqdm
from tools.colormap import colormap
import opts
np.bool = np.bool_

# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()

def main(args):
    print("Inference only supports for batch size = 1")
    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    utils.init_distributed_mode(args)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # save path
    output_dir = args.output_dir
    args.log_file = join(args.output_dir, 'log.txt')
    with open(args.log_file, 'w') as fp:
        fp.writelines(" ".join(sys.argv) + '\n')
        fp.writelines(str(args.__dict__) + '\n\n')

    start_time = time.time()
    # model
    model = build_samwise(args)
    device = torch.device(args.device)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params:', n_parameters)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if list(checkpoint['model'].keys())[0].startswith('module'):
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        checkpoint = on_load_checkpoint(model_without_ddp, checkpoint)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

    print('Start inference')
    eval_davis(args, model, output_dir)

    end_time = time.time()
    total_time = end_time - start_time

    print("Total inference time: %.4f s" % (total_time))



def eval_davis(args, model, save_path_prefix):
    print("Inference only supports for batch size = 1")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # save path
    save_path_prefix = os.path.join(save_path_prefix, "eval_davis", args.split)
    os.makedirs(save_path_prefix, exist_ok=True)

    # load data
    root = Path(args.davis_path)  # data/ref-davis
    img_folder = os.path.join(root, args.split, "JPEGImages")
    meta_file = os.path.join(root, "meta_expressions", args.split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    video_list = list(data.keys())

    start_time = time.time()
    print('Start inference')
    sub_video_list = video_list

    sub_processor(args, model, data, save_path_prefix, img_folder, sub_video_list)

    for annotator in range(4):
        args.results_path = os.path.join(save_path_prefix, f"anno_{annotator}")
        eval_davis_compute_metrics(args)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total inference time: {total_time:.2f} s")


def sub_processor(args, model, data, save_path_prefix, img_folder, video_list):
    progress = tqdm(
            total=len(video_list),
            ncols=0
        )

    # get palette
    palette_img = os.path.join(args.davis_path, "valid/Annotations/blackswan/00000.png")
    palette = Image.open(palette_img).getpalette()

    # start inference
    model.eval()

    # 1. for each video
    for video in video_list:
        metas = []

        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]  # start from 0
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas

        # since there are 4 annotations
        num_obj = num_expressions // 4

        # 2. for each annotator
        for anno_id in range(4):  # 4 annotators
            anno_masks = []  # [num_obj+1, video_len, h, w], +1 for background

            for obj_id in range(num_obj):
                i = obj_id * 4 + anno_id
                video_name = meta[i]["video"]
                exp = meta[i]["exp"]
                frames = meta[i]["frames"]

                all_pred_masks = []

                vd = VideoEvalDataset(join(img_folder, video_name), frames, max_size=args.max_size)
                dl = DataLoader(vd, batch_size=args.eval_clip_window,
                                num_workers=args.num_workers, shuffle=False)
                origin_w, origin_h = vd.origin_w, vd.origin_h
                # 3. for each clip
                for imgs, clip_frames_ids in dl:
                    clip_frames_ids = clip_frames_ids.tolist()
                    img_h, img_w = imgs.shape[-2:]
                    imgs = imgs.to(args.device)
                    size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
                    target = {"size": size, 'frame_ids': clip_frames_ids}
                    with torch.no_grad():
                        outputs = model([imgs], [exp], [target])

                    pred_masks = outputs["pred_masks"]  # [t, q, h, w]
                    pred_masks = pred_masks.unsqueeze(0)

                    pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear',
                                               align_corners=False)
                    pred_masks = pred_masks.sigmoid()[0]  # [t, h, w], NOTE: here mask is score
                    all_pred_masks.append(pred_masks)

                all_pred_masks = torch.cat(all_pred_masks, dim=0)  # (video_len, h, w)
                anno_masks.append(all_pred_masks)

            # handle a complete image (all objects of a annotator)
            anno_masks = torch.stack(anno_masks)  # [num_obj, video_len, h, w]
            t, h, w = anno_masks.shape[-3:]
            anno_masks[anno_masks < 0.5] = 0.0
            background = 0.1 * torch.ones(1, t, h, w).to(args.device)
            anno_masks = torch.cat([background, anno_masks], dim=0)  # [num_obj+1, video_len, h, w]
            out_masks = torch.argmax(anno_masks, dim=0)  # int, the value indicate which object, [video_len, h, w]

            out_masks = out_masks.detach().cpu().numpy().astype(np.uint8)  # [video_len, h, w]

            # save results
            anno_save_path = os.path.join(save_path_prefix, f"anno_{anno_id}", video)
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            os.makedirs(anno_save_path, exist_ok=True)
            for f in range(out_masks.shape[0]):
                img_E = Image.fromarray(out_masks[f])
                img_E.putpalette(palette)
                if utils.is_main_process():
                    img_E.save(os.path.join(anno_save_path, '{:05d}.png'.format(f)))
        progress.update(1)

def eval_davis_compute_metrics(args):
    time_start = time.time()
    csv_name_global = f'global_results-{args.set}.csv'
    csv_name_per_sequence = f'per-sequence_results-{args.set}.csv'
    args.log_file = join(args.output_dir, 'log.txt')
    print(f'using {args.results_path}')
    # Check if the method has been evaluated before, if so read the results, otherwise compute the results
    csv_name_global_path = os.path.join(args.results_path, csv_name_global)
    csv_name_per_sequence_path = os.path.join(args.results_path, csv_name_per_sequence)
    if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
        print('Using precomputed results...')
        table_g = pd.read_csv(csv_name_global_path)
        table_seq = pd.read_csv(csv_name_per_sequence_path)
    else:
        print(f'Evaluating sequences for the {args.task} task...')
        # Create dataset and evaluate
        dataset_eval = DAVISEvaluation(davis_root=args.davis_path + "/DAVIS", task=args.task, gt_set=args.set)
        metrics_res = dataset_eval.evaluate(args.results_path)
        J, F = metrics_res['J'], metrics_res['F']

        # Generate dataframe for the general results
        g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
        final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
        g_res = np.array(
            [final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
             np.mean(F["D"])])
        g_res = np.reshape(g_res, [1, len(g_res)])
        table_g = pd.DataFrame(data=g_res, columns=g_measures)
        with open(csv_name_global_path, 'w') as f:
            table_g.to_csv(f, index=False, float_format="%.5f")
        print(f'Global results saved in {csv_name_global_path}')

        # Generate a dataframe for the per sequence results
        seq_names = list(J['M_per_object'].keys())
        seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
        J_per_object = [J['M_per_object'][x] for x in seq_names]
        F_per_object = [F['M_per_object'][x] for x in seq_names]
        table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
        if utils.is_main_process():
            with open(csv_name_per_sequence_path, 'w') as f:
                table_seq.to_csv(f, index=False, float_format="%.5f")
        print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

    # Print the results
    sys.stdout.write(f"--------------------------- Global results for {args.set} ---------------------------\n")
    print(table_g.to_string(index=False))
    sys.stdout.write(f"\n---------- Per sequence results for {args.set} ----------\n")
    print(table_seq.to_string(index=False))
    total_time = time.time() - time_start
    sys.stdout.write('\nTotal time:' + str(total_time))
    if utils.get_rank() == 0:
        with open(args.log_file, 'a') as fp:
            fp.write(table_seq.to_string(index=False) + '\n')
            fp.write(str(g_measures) + '\n')
            fp.write(str(g_res) + '\n\n')


if __name__ == '__main__':
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser('SAMWISE evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    name_exp = args.name_exp
    args.output_dir = os.path.join(args.output_dir, name_exp)


    os.makedirs(args.output_dir, exist_ok=True)

    main(args)