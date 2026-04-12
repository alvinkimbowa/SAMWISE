"""
Training script of SAMWISE
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from util.misc import on_load_checkpoint
import datasets.samplers as samplers
from datasets import build_dataset
from engine import train_one_epoch
from models.samwise import build_samwise
from os.path import join
import sys
import opts


def main(args):
    utils.init_distributed_mode(args)
    if args.output_dir and utils.get_rank() == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args.log_file = join(args.output_dir, 'log.txt')
        with open(args.log_file, 'w') as fp:
            fp.writelines(" ".join(sys.argv)+'\n')
            fp.writelines(str(args.__dict__)+'\n\n')

    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = build_samwise(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    n_parameters_tot = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters_tot)

    head = []
    fix = []
    for k, v in model_without_ddp.named_parameters():
        if v.requires_grad:
            head.append(v)
        else:
            fix.append(v)

    print("Trainable parameters: ", sum(p.numel() for p in head))
    print("Parameters fixed: ", sum(p.numel() for p in fix))

    param_list = [{
        'params': head,
        'initial_lr': args.lr
    }]

    optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(args.dataset_file, image_set="train", args=args)

    args.batch_size = int(args.batch_size / args.ngpu)
    if args.distributed:
        sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        checkpoint = on_load_checkpoint(model_without_ddp, checkpoint)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if args.resume_optimizer and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print(
                    'Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1


    if args.motion_prompt:
        # initialize projection weights for the MOTION prompt with those of CLS
        mlp_weights = model_without_ddp.sam.sam_prompt_encoder.project_text.state_dict()
        model_without_ddp.sam.sam_prompt_encoder.project_motion_prompts.load_state_dict(mlp_weights)

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
                    model, data_loader_train, optimizer, device, epoch,
                    args.clip_max_norm, lr_scheduler=lr_scheduler, args=args)

        if args.output_dir:
            print("Save Model")
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters_tot,
        }

        if args.dataset_file == 'ytvos':
            if os.path.abspath(args.ytvos_path) == os.path.abspath('data/ref-youtube-vos'):
                print("Evaluate on DAVIS: ")
                from inference_davis import eval_davis
                m = model
                out_dir = join(args.output_dir, f'valid_epoch{str(epoch).zfill(2)}')
                eval_davis(args, m, out_dir)
            else:
                print("Evaluate on Echo validation split: ")
                from inference_echo import eval_echo
                m = model
                out_dir = join(args.output_dir, f'valid_epoch{str(epoch).zfill(2)}')
                result = eval_echo(args, m, out_dir)
                if utils.is_main_process():
                    out_str = f'Epoch: {epoch}:\nJ: {result[0]},\t F: {result[1]},\t J&F: {result[2]}'
                    print(out_str)
                    log_stats.update({'val_J': result[0], 'val_F': result[1], 'val_J&F': result[2]})
                    with (output_dir / "log.txt").open("a") as f:
                        f.write(out_str + "\n")
        elif args.dataset_file == 'mevis':
            m = model
            print("Evaluate on MeVis: ")
            from inference_mevis import eval_mevis
            out_dir = join(args.output_dir, f'valid_epoch{str(epoch).zfill(2)}')
            args.split = 'valid_u'
            result = eval_mevis(args, m, out_dir, out_dir)

            if utils.is_main_process():
                out_str = f'Epoch: {epoch}:\nJ: {result[0]},\t F: {result[1]},\t J&F: {result[2]}'
                print(out_str)
                with (output_dir / "log.txt").open("a") as f:
                    f.write(out_str + "\n")

        if utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser('SAMWISE training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    name_exp = args.name_exp
    args.output_dir = os.path.join(args.output_dir, name_exp)

    main(args)

