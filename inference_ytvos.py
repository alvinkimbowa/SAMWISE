'''
Inference code for SAMWISE, on Ref-Youtube-VOS
Modified from DETR (https://github.com/facebookresearch/detr)
'''
from datasets.transform_utils import vis_add_mask
import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
import util.misc as utils
import os
from PIL import Image
import torch.nn.functional as F
import json
from tqdm import tqdm
from os.path import join
import sys
from tools.colormap import colormap
import opts
from models.samwise import build_samwise
from util.misc import on_load_checkpoint
from datasets.transform_utils import VideoEvalDataset
from torch.utils.data import DataLoader

# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()


def main(args):
    args.batch_size = 1
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

    split = args.split
    # save path
    output_dir = args.output_dir
    save_path_prefix = os.path.join(output_dir, 'Annotations')

    os.makedirs(save_path_prefix, exist_ok=True)
    args.log_file = join(args.output_dir, 'log.txt')
    with open(args.log_file, 'w') as fp:
        fp.writelines(" ".join(sys.argv)+'\n')
        fp.writelines(str(args.__dict__)+'\n\n')        

    save_visualize_path_prefix = os.path.join(output_dir, split + '_images')
    if args.visualize:
        os.makedirs(save_visualize_path_prefix, exist_ok=True)

    # load data
    root = Path(args.ytvos_path) # data/ref-youtube-vos
    img_folder = os.path.join(root, split, "JPEGImages")
    meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    valid_test_videos = set(data.keys())
    # for some reasons the competition's validation expressions dict contains both the validation (202) & 
    # test videos (305). so we simply load the test expressions dict and use it to filter out the test videos from
    # the validation expressions dict:
    test_meta_file = os.path.join(root, "meta_expressions", "test", "meta_expressions.json")
    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())
    valid_videos = valid_test_videos - test_videos
    video_list = sorted([video for video in valid_videos])
    assert len(video_list) == 202, f'error: {len(video_list)} incorrect number of validation videos'

    start_time = time.time()
    print('Start inference')
        
    result_dict = sub_processor(args, data, save_path_prefix, save_visualize_path_prefix, 
                    img_folder, video_list)

    end_time = time.time()
    total_time = end_time - start_time

    result_dict = dict(result_dict)
    num_all_frames_gpus = 0
    for pid, num_all_frames in result_dict.items():
        num_all_frames_gpus += num_all_frames

    print("Total inference time: %.4f s" %(total_time))

def sub_processor(args, data, save_path_prefix, save_visualize_path_prefix, img_folder, video_list):
    result_dict = dict()
    progress = tqdm(
        total=len(video_list),
        ncols=0
    )

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
        checkpoint = on_load_checkpoint(model_without_ddp, checkpoint)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
    else:
        raise ValueError('Please specify the checkpoint for inference.')


    # start inference
    num_all_frames = 0 
    model.eval()

    # 1. For each video
    for video in video_list:
        metas = [] # list[dict], length is number of expressions

        expressions = data[video]["expressions"]   
        expression_list = list(expressions.keys()) 
        num_expressions = len(expression_list)
        video_len = len(data[video]["frames"])

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas

        # 2. For each expression
        for i in range(num_expressions):
            video_name = meta[i]["video"]
            exp = meta[i]["exp"]
            exp_id = meta[i]["exp_id"]
            frames = meta[i]["frames"]

            video_len = len(frames)
            save_path = os.path.join(save_path_prefix, video_name, exp_id)
            os.makedirs(save_path, exist_ok=True)
            expected_mask_files = [os.path.join(save_path, frame + ".png") for frame in frames]
            if all(os.path.exists(mask_file) for mask_file in expected_mask_files):
                continue
            missing_frame_indices = [idx for idx, mask_file in enumerate(expected_mask_files) if not os.path.exists(mask_file)]
            last_missing_idx = missing_frame_indices[-1]

            # 3. for each clip
            vd = VideoEvalDataset(join(img_folder, video_name), frames, max_size=args.max_size)
            dl = DataLoader(vd, batch_size=args.eval_clip_window,
                    num_workers=args.num_workers, shuffle=False)
            origin_w, origin_h = vd.origin_w, vd.origin_h
            # 3. for each clip
            for imgs, clip_frames_ids in dl:
                clip_frames_ids = clip_frames_ids.tolist()
                clip_frames = [frames[idx] for idx in clip_frames_ids]
                clip_mask_files = [os.path.join(save_path, frame + ".png") for frame in clip_frames]
                if clip_frames_ids[0] > last_missing_idx:
                    break

                imgs = imgs.to(args.device)  # [eval_clip_window, 3, h, w]
                img_h, img_w = imgs.shape[-2:]
                size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
                target = {"size": size, 'frame_ids': clip_frames_ids}

                with torch.no_grad():
                    outputs = model([imgs], [exp], [target])

                pred_masks = outputs["pred_masks"]  # [t, q, h, w]
                pred_masks = pred_masks.unsqueeze(0)
                pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False) 
                pred_masks = (pred_masks.sigmoid() > args.threshold)[0].cpu() 
                for frame_name, clip_mask_file, pred_mask in zip(clip_frames, clip_mask_files, pred_masks):
                    pred_mask = pred_mask.numpy().astype(np.float32)
                    if not os.path.exists(clip_mask_file):
                        mask = Image.fromarray(pred_mask * 255).convert('L')
                        mask.save(clip_mask_file)

                    if args.visualize:
                        save_visualize_path_dir = os.path.join(save_visualize_path_prefix, video, str(i))
                        os.makedirs(save_visualize_path_dir, exist_ok=True)
                        save_visualize_path = os.path.join(save_visualize_path_dir, frame_name + '.png')
                        if not os.path.exists(save_visualize_path):
                            img_path = os.path.join(img_folder, video_name, frame_name + '.jpg')
                            source_img = Image.open(img_path).convert('RGBA')
                            source_img = vis_add_mask(source_img, pred_mask, color_list[i%len(color_list)])
                            source_img.save(save_visualize_path)
        progress.update(1)

    result_dict["0"] = num_all_frames

    return result_dict


# visuaize functions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# Visualization functions
def draw_reference_points(draw, reference_points, img_size, color):
    W, H = img_size
    for i, ref_point in enumerate(reference_points):
        init_x, init_y = ref_point
        x, y = W * init_x, H * init_y
        cur_color = color
        draw.line((x-10, y, x+10, y), tuple(cur_color), width=4)
        draw.line((x, y-10, x, y+10), tuple(cur_color), width=4)

def draw_sample_points(draw, sample_points, img_size, color_list):
    alpha = 255
    for i, samples in enumerate(sample_points):
        for sample in samples:
            x, y = sample
            cur_color = color_list[i % len(color_list)][::-1]
            cur_color += [alpha]
            draw.ellipse((x-2, y-2, x+2, y+2), 
                            fill=tuple(cur_color), outline=tuple(cur_color), width=1)

def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8') # np
    mask = mask > 0.5

    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img = Image.fromarray(origin_img)
    return origin_img



if __name__ == '__main__':
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser('SAMWISE evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    name_exp = args.name_exp
    args.output_dir = os.path.join(args.output_dir, name_exp)

    main(args)
