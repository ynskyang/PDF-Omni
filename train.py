from __future__ import print_function, division

import json
import os
import os.path as osp
import sys
import time
from datetime import datetime
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.amp import GradScaler

from dataset import Dataset, MultiDataset
from utils.common import *
from module.network import PDFOmni
from module.loss_functions import sequence_loss, lec_loss

torch.backends.cudnn.benchmark = True

parser = ArgumentParser(description='Training for PDF-Omni')

parser.add_argument('--name', default='PDFOmni', help="name of your experiment")
parser.add_argument('--restore_ckpt', help="restore checkpoint")
parser.add_argument('--pretrain_ckpt', help="pretrained checkpoint for finetuning")

parser.add_argument('--db_root', default='../omnidata', type=str, help='path to dataset')
parser.add_argument('--dbname', nargs='+', default=['omnithings'], type=str,
                    choices=['omnithings', 'omnihouse', 'sunny', 'cloudy', 'sunset'],
                    help='databases to train')

parser.add_argument('--phi_deg', type=float, default=45.0, help='phi_deg')
parser.add_argument('--num_invdepth', type=int, default=192, metavar='N', help='number of disparity')
parser.add_argument('--equirect_size', type=int, nargs='+', default=[160, 640], help="size of out ERP.")
parser.add_argument('--use_rgb', action='store_true', help='use 3-channel rgb color images as input')

parser.add_argument('--base_channel', type=int, default=32, choices=[32], help='base channel of the network')
parser.add_argument('--encoder_downsample_twice', action='store_true',
                    help='the feature extractor downsamples fisheye input twice instead of once.')
parser.add_argument('--num_downsample', type=int, default=1, help="resolution of the disparity field (1/2^K)")
parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
parser.add_argument('--gru_small_kernel', type=int, nargs='+', default=[1],
                    help='small GRU kernel size (1 or 2 ints). e.g., "1" or "1 1"')
parser.add_argument('--gru_large_kernel', type=int, nargs='+', default=[7],
                    help='large GRU kernel size (1 or 2 ints). e.g., "7" or "1 7"')
parser.add_argument('--gru_large_mode', type=str, default='sep', choices=['sep', 'conv'],
                    help='sep: 1xk then kx1 (separable), conv: standard conv (square/rect)')

parser.add_argument('--gwc_groups', type=int, default=8, help='num groups for GWC (attention branch)')
parser.add_argument('--acv_concat_ch', type=int, default=16, help='per-side compressed channels for concat volume')
parser.add_argument('--geo_ch', type=int, default=8, help='hourglass output channels for geo encoding volume')
parser.add_argument('--use_geo_acv', action='store_true', help='inject spherical coord channels into ACV attention')

parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--fix_bn', action='store_true', help='fix batch normalization')

parser.add_argument('--use_ge', action='store_true', help="Enable GE (geometry embedding)")
parser.add_argument('--use_digru', action='store_true', help="Enable DiGRU (Distortion-informed GRU)")

parser.add_argument('--ge_scale', type=float, default=1.0, help='Gaussian scale for GE Fourier features')
parser.add_argument('--ge_map_size', type=int, default=256, help='Mapping size for GE Fourier features')

parser.add_argument('--total_epochs', type=int, default=30, help='total epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--train_iters', type=int, default=12,
                    help="number of updates to the disparity field in each forward pass.")
parser.add_argument('--valid_iters', type=int, default=12,
                    help='number of flow-field updates during validation forward pass')
parser.add_argument('--lr', type=float, default=0.0005, help="max learning rate.")
parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")
parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping (global norm)')

parser.add_argument('--ot_iter', type=int, default=3, help='Sinkhorn iterations for OT init in volume generator')

parser.add_argument('--lec_weight', type=float, default=0.1, help='LEC loss weight (set >0 to enable)')

parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes')
parser.add_argument('--log_every', type=int, default=100, help='log/vis frequency in steps')
parser.add_argument('--val_stride', type=int, default=1, help='evaluate every Nth val frame per epoch; final metrics should come from eval.py on the full test set')

args = parser.parse_args()

if args.name == parser.get_default('name'):
    args.name = datetime.now().strftime("%y%m%d_%H%M")
else:
    args.name = '%s_%s' % (args.name, datetime.now().strftime("%y%m%d_%H%M"))


def _parse_kernel(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return int(value[0])
        if len(value) == 2:
            return (int(value[0]), int(value[1]))
        raise ValueError("kernel size expects 1 or 2 ints")
    return int(value)

opts = Edict()
opts.name = args.name
opts.model_dir = os.path.join('./checkpoints', args.name)
opts.runs_dir = os.path.join('./runs', args.name)

opts.snapshot_path = args.restore_ckpt
opts.pretrain_path = args.pretrain_ckpt
opts.dbname = args.dbname
opts.db_root = args.db_root

opts.data_opts = Edict()
if args.phi_deg != parser.get_default('phi_deg'):
    opts.data_opts.phi_deg = args.phi_deg
opts.data_opts.num_invdepth = args.num_invdepth
opts.data_opts.equirect_size = args.equirect_size
opts.data_opts.num_downsample = args.num_downsample
opts.data_opts.use_rgb = args.use_rgb

opts.net_opts = Edict()
opts.net_opts.base_channel = args.base_channel
opts.net_opts.num_invdepth = opts.data_opts.num_invdepth
opts.net_opts.use_rgb = opts.data_opts.use_rgb
opts.net_opts.encoder_downsample_twice = args.encoder_downsample_twice
opts.net_opts.num_downsample = args.num_downsample
opts.net_opts.corr_levels = args.corr_levels
opts.net_opts.corr_radius = args.corr_radius
opts.net_opts.mixed_precision = args.mixed_precision
opts.net_opts.fix_bn = args.fix_bn
opts.net_opts.gru_small_kernel = _parse_kernel(args.gru_small_kernel)
opts.net_opts.gru_large_kernel = _parse_kernel(args.gru_large_kernel)
opts.net_opts.gru_large_mode = args.gru_large_mode
opts.net_opts.gwc_groups = args.gwc_groups
opts.net_opts.acv_concat_ch = args.acv_concat_ch
opts.net_opts.geo_ch = args.geo_ch
opts.net_opts.ot_iter = args.ot_iter
opts.net_opts.pe_mode = "grid"
opts.net_opts.use_geo_acv = args.use_geo_acv
opts.net_opts.use_ge = args.use_ge
opts.net_opts.use_digru = args.use_digru
opts.net_opts.distortion_mode = "hyperbolic_dual_disk"
opts.net_opts.ge_scale = args.ge_scale
opts.net_opts.ge_map_size = args.ge_map_size

opts.total_epochs = args.total_epochs
opts.batch_size = args.batch_size
opts.train_iters = args.train_iters
opts.valid_iters = args.valid_iters
opts.lr = args.lr
opts.wdecay = args.wdecay
opts.grad_clip = args.grad_clip
opts.lec_weight = args.lec_weight

opts.num_workers = args.num_workers
opts.log_every = args.log_every

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(model, num_steps):
    optimizer = optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.wdecay, eps=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, opts.lr, num_steps + 100,
        pct_start=0.01, cycle_momentum=False, anneal_strategy='cos'
    )
    return optimizer, scheduler


def require_checkpoint(path, label):
    if path is not None and not osp.isfile(path):
        raise SystemExit(f'{label} not found: {path}')


def train(epoch_total, load_state):
    if len(opts.dbname) > 1:
        data = MultiDataset(opts.dbname, opts.data_opts, db_root=opts.db_root)
    else:
        data = Dataset(opts.dbname[0], opts.data_opts, db_root=opts.db_root)

    dbloader = torch.utils.data.DataLoader(
        data, batch_size=opts.batch_size,
        pin_memory=True, shuffle=True,
        num_workers=opts.num_workers, drop_last=True
    )
    total_num_steps = len(data) * opts.total_epochs // max(opts.batch_size, 1)

    net = nn.DataParallel(PDFOmni(opts.net_opts)).cuda()
    if opts.net_opts.fix_bn:
        net.module.freeze_bn()
    param_count = count_parameters(net)
    LOG_INFO("Parameter Count: %d" % param_count)

    optimizer, scheduler = fetch_optimizer(net, total_num_steps)
    scaler = GradScaler('cuda', enabled=opts.net_opts.mixed_precision)

    os.makedirs(opts.model_dir, exist_ok=True)
    LOG_INFO('"%s" directory created' % (opts.model_dir))
    os.makedirs(opts.runs_dir, exist_ok=True)
    LOG_INFO('"%s" directory created' % (opts.runs_dir))
    args_path = osp.join(opts.model_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump({
            'cmd': ' '.join(sys.argv),
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'args': vars(args),
        }, f, indent=2, default=str)
    LOG_INFO('run config saved to "%s"' % args_path)
    writer = SummaryWriter(log_dir=opts.runs_dir)

    LOG_INFO(
        f"[Setup] name={opts.name} | epochs={opts.total_epochs} | batch_size={opts.batch_size} "
        f"| num_workers={opts.num_workers} | log_every={opts.log_every} | mp={opts.net_opts.mixed_precision} "
        f"| use_geo_acv={opts.net_opts.use_geo_acv} "
        f"| distortion_mode={opts.net_opts.distortion_mode}"
    )

    start_epoch = 0
    if load_state:
        if opts.snapshot_path:
            snapshot = torch.load(opts.snapshot_path, weights_only=False)
            if 'net_state_dict' in snapshot.keys():
                net.load_state_dict(snapshot['net_state_dict'])
                LOG_INFO('checkpoint %s is loaded' % (opts.snapshot_path))
            if 'epoch' in snapshot.keys():
                start_epoch = snapshot['epoch'] + 1
            if 'epoch_loss' in snapshot.keys():
                epoch_loss = snapshot['epoch_loss']
            if 'optimizer' in snapshot.keys():
                optimizer.load_state_dict(snapshot['optimizer'])
            if 'scheduler' in snapshot.keys():
                scheduler.load_state_dict(snapshot['scheduler'])
            if 'epoch' in snapshot.keys() and 'epoch_loss' in snapshot.keys():
                LOG_INFO('startepoch:%d epoch_loss:%f' % (start_epoch, epoch_loss))

        if opts.pretrain_path:
            snapshot = torch.load(opts.pretrain_path, weights_only=False)
            if 'net_state_dict' in snapshot.keys():
                net.load_state_dict(snapshot['net_state_dict'])
                LOG_INFO('checkpoint %s is loaded' % (opts.pretrain_path))

    grids = [torch.tensor(grid, requires_grad=False).cuda() for grid in data.grids]

    total_iters = len(data) * start_epoch // max(opts.batch_size, 1)
    LOG_EVERY = max(1, int(opts.log_every))

    for epoch in range(start_epoch, epoch_total):
        net.train()
        train_loss = 0.0
        epoch_loss = 0.0
        LOG_INFO('\nEpoch: %d' % epoch)

        for step, data_blob in enumerate(dbloader):
            start_time = time.time()
            imgs, gt, valid, raw_imgs = data_blob

            imgs = [img.cuda(non_blocking=True) for img in imgs]
            valid = valid.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            out = net(imgs, grids, opts.train_iters)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                predictions, init_prob = out
            else:
                predictions, init_prob = out, None

            loss_seq = sequence_loss(predictions, gt.unsqueeze(1), valid.unsqueeze(1))
            loss = loss_seq
            loss_lec = None
            lec_stats = None
            if init_prob is not None and opts.lec_weight and opts.lec_weight > 0:
                loss_lec, lec_stats = lec_loss(
                    prob_volume=init_prob,
                    gt_disp=gt.unsqueeze(1),
                    img=imgs[0],
                    valid_mask=valid.unsqueeze(1),
                    full_num_invdepth=opts.net_opts.num_invdepth,
                    return_stats=True,
                )
                loss = loss + float(opts.lec_weight) * loss_lec

            train_loss += float(loss.detach().cpu())
            epoch_loss = train_loss / (step + 1)

            if step % LOG_EVERY == 0:
                LOG_INFO(
                    "Iter %d training loss = %.4f, seq = %.4f, lec = %.4f, avg = %.4f, time = %.2f"
                    % (
                        total_iters,
                        float(loss.detach().cpu()),
                        float(loss_seq.detach().cpu()),
                        float(loss_lec.detach().cpu()) if loss_lec is not None else 0.0,
                        float(epoch_loss),
                        time.time() - start_time,
                    )
                )
                invdepth_idx_vis = torch.clamp(
                    predictions[-1][0][0], 0, opts.net_opts.num_invdepth - 1
                )
                writer.add_scalar("train/loss", float(loss.detach().cpu()), total_iters)
                writer.add_scalar("train/loss_seq", float(loss_seq.detach().cpu()), total_iters)
                if loss_lec is not None:
                    writer.add_scalar("train/loss_lec", float(loss_lec.detach().cpu()), total_iters)
                    writer.add_scalar(
                        "train/loss_lec_weighted",
                        float((float(opts.lec_weight) * loss_lec).detach().cpu()),
                        total_iters,
                    )
                    if lec_stats is not None:
                        writer.add_scalar("train/lec_entropy", float(lec_stats["entropy"].cpu()), total_iters)
                        writer.add_scalar("train/lec_radius", float(lec_stats["radius"].cpu()), total_iters)
                        writer.add_scalar("train/lec_mass", float(lec_stats["mass"].cpu()), total_iters)
                        writer.add_scalar(
                            "train/lec_gt_to_volume_scale",
                            float(lec_stats["gt_to_volume_scale"].cpu()),
                            total_iters,
                        )
                writer.add_scalar("train/epoch_loss", float(epoch_loss), total_iters)
                invdepth_vis = data.indexToInvdepth(toNumpy(invdepth_idx_vis))
                raw_imgs_np = [toNumpy(raw[0]) for raw in raw_imgs]
                vis_img = data.makeVisImage(raw_imgs_np, invdepth_vis, gt=toNumpy(gt[0]))
                writer.add_image("train/vis", vis_img.transpose(2, 0, 1), total_iters)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if opts.grad_clip is not None and opts.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), opts.grad_clip)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before_step:
                scheduler.step()

            total_iters += 1

        invdepth_idx = torch.clamp(predictions[-1][0][0], 0, opts.net_opts.num_invdepth - 1)
        writer.add_scalar("train/epoch_loss", float(epoch_loss), total_iters)
        invdepth = data.indexToInvdepth(toNumpy(invdepth_idx))
        raw_imgs_np = [toNumpy(raw[0]) for raw in raw_imgs]
        vis_img = data.makeVisImage(raw_imgs_np, invdepth, gt=toNumpy(gt[0]))
        writer.add_image("train/vis", vis_img.transpose(2, 0, 1), total_iters)

        net.eval()
        eval_list = data.opts.test_idx[::max(1, args.val_stride)]
        errors = np.zeros((len(eval_list), 5))
        for d in range(len(eval_list)):
            fidx = eval_list[d]
            imgs_eval, gt_eval, valid_eval, raw_imgs_eval = data.loadSample(fidx)
            imgs_eval = [torch.Tensor(img).unsqueeze(0).cuda() for img in imgs_eval]
            with torch.no_grad():
                invdepth_idx_eval = net(imgs_eval, grids, opts.valid_iters, test_mode=True)
            invdepth_idx_eval = toNumpy(invdepth_idx_eval[0, 0])
            errors[d, :] = data.evalError(invdepth_idx_eval, gt_eval, valid_eval)

        invdepth_eval = data.indexToInvdepth(invdepth_idx_eval)
        raw_imgs_eval_np = [toNumpy(raw) for raw in raw_imgs_eval]
        vis_img_eval = data.makeVisImage(raw_imgs_eval_np, invdepth_eval, gt=toNumpy(gt_eval))
        writer.add_image("val/vis", vis_img_eval.transpose(2, 0, 1), total_iters)

        mean_errors = errors.mean(axis=0)
        writer.add_scalar("val/>1",  float(mean_errors[0]), total_iters)
        writer.add_scalar("val/>3",  float(mean_errors[1]), total_iters)
        writer.add_scalar("val/>5",  float(mean_errors[2]), total_iters)
        writer.add_scalar("val/MAE", float(mean_errors[3]), total_iters)
        writer.add_scalar("val/RMS", float(mean_errors[4]), total_iters)
        LOG_INFO('>1: %.3f, >3: %.3f, >5: %.3f, MAE: %.3f, RMS: %.3f' %
                 (mean_errors[0], mean_errors[1], mean_errors[2], mean_errors[3], mean_errors[4]))

        # persist per-epoch validation record (valid_record.txt + val_history.json)
        import json as _json
        _rec = {'epoch': int(epoch), 'train_loss': float(epoch_loss),
                '>1': float(mean_errors[0]), '>3': float(mean_errors[1]),
                '>5': float(mean_errors[2]), 'MAE': float(mean_errors[3]),
                'RMS': float(mean_errors[4])}
        with open(osp.join(opts.model_dir, 'valid_record.txt'), 'a') as _f:
            _f.write('epoch: %d\ntrain_loss: %.4f\n' % (epoch, epoch_loss))
            _f.write('>1: %.3f, >3: %.3f, >5: %.3f, MAE: %.3f, RMS: %.3f\n' % tuple(mean_errors[:5]))
            _f.write('=' * 48 + '\n')
        _hp = osp.join(opts.model_dir, 'val_history.json')
        try:
            _hist = _json.load(open(_hp))
        except Exception:
            _hist = []
        _hist = [h for h in _hist if h.get('epoch') != int(epoch)] + [_rec]
        _json.dump(sorted(_hist, key=lambda h: h['epoch']), open(_hp, 'w'), indent=2)

        savefilename = osp.join(opts.model_dir, '%s_e%d.pth' % (opts.name, epoch))
        torch.save({
                'net_state_dict': net.state_dict(),
                'net_opts': opts.net_opts,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch_loss': float(epoch_loss),
            }, savefilename)


def main():
    load_state = opts.snapshot_path is not None or opts.pretrain_path is not None
    require_checkpoint(opts.snapshot_path, '--restore_ckpt')
    require_checkpoint(opts.pretrain_path, '--pretrain_ckpt')
    train(opts.total_epochs, load_state)


if __name__ == "__main__":
    main()
