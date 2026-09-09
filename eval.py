from __future__ import print_function, division

from argparse import ArgumentParser
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch

from dataset import Dataset
from utils.common import *
from utils.image import *
from module.network import PDFOmni

try:
    import tifffile
except Exception:
    tifffile = None

torch.backends.cudnn.benchmark = True

parser = ArgumentParser(description='Evaluation for PDF-Omni')
parser.add_argument('--name', default='PDFOmni', help="name of your experiment")
parser.add_argument('--restore_ckpt', required=True, help="restore checkpoint")
parser.add_argument('--output_dir', default='./results', type=str, help='directory to save evaluation outputs')

parser.add_argument('--db_root', default='../omnidata', type=str, help='path to dataset')
parser.add_argument('--dbname', default='omnithings', type=str,
                    choices=['omnithings', 'omnihouse', 'sunny', 'cloudy', 'sunset', 'all'],
                    help='databases to evaluation')

parser.add_argument('--phi_deg', type=float, default=45.0, help='phi_deg')
parser.add_argument('--equirect_size', type=int, nargs='+', default=[160, 640], help="size of out ERP.")

parser.add_argument('--valid_iters', type=int, default=12,
                    help='number of flow-field updates during validation forward pass')

parser.add_argument('--vis', action='store_true', help='oneline visualization')
parser.add_argument('--save_result', action='store_true', help='save inverse depth prediction results (float32 TIFF)')
parser.add_argument('--save_misc', action='store_true', help='save misc visualizations (includes colorized invdepth)')
parser.add_argument('--save_point_cloud', action='store_true', help='save point cloud')
parser.add_argument('--no_save_float', action='store_true', help='do NOT save float32 TIFF even if --save_result is set')

args = parser.parse_args()

opts = Edict()
opts.snapshot_path = args.restore_ckpt
opts.name = args.name
opts.output_dir = args.output_dir

opts.db_root = args.db_root

opts.data_opts = Edict()
opts.data_opts.color_aug = False
if args.phi_deg != parser.get_default('phi_deg'):
    opts.data_opts.phi_deg = args.phi_deg
opts.data_opts.equirect_size = args.equirect_size

opts.valid_iters = args.valid_iters
opts.net_opts = Edict()

opts.vis = args.vis
opts.save_result = args.save_result
opts.save_misc = args.save_misc
opts.save_point_cloud = args.save_point_cloud
opts.save_float = (not args.no_save_float)

if opts.vis:
    fig = plt.figure(frameon=False, figsize=(25, 10), dpi=40)
    plt.ion()
    plt.show()
else:
    fig = None


def _save_float_tiff(invdepth_float: np.ndarray, out_path: str):
    if tifffile is None:
        LOG_INFO(f"tifffile not available; skipping float TIFF save for {out_path}")
        return

    img = np.asarray(invdepth_float, dtype=np.float32)
    out_dir = osp.dirname(out_path)
    if out_dir and not osp.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
        tifffile.imwrite(
            out_path,
            img,
            dtype=np.float32,
            photometric='minisblack',
            planarconfig='contig',
            metadata=None,
            compression='deflate',
            compressionlevel=6,
        )
    except TypeError:
        tifffile.imwrite(
            out_path,
            img,
            dtype=np.float32,
            photometric='minisblack',
            planarconfig='contig',
            metadata=None,
            compression='deflate',
            compressionargs={"level": 6},
        )


def evaluate_dataset(opts, net, dbname):
    LOG_INFO(f"=== Start Evaluation for: {dbname} ===")

    opts.dbname = dbname
    opts.result_dir = osp.join(opts.output_dir, opts.dbname)
    snapshot_name = osp.splitext(osp.basename(opts.snapshot_path))[0]

    opts.out_invdepth_fmt = osp.join(opts.result_dir, 'invdepth_%s_' + snapshot_name + '.tiff')
    opts.out_misc_fmt     = osp.join(opts.result_dir, 'misc_%s_'     + snapshot_name + '.png')
    opts.out_point_fmt    = osp.join(opts.result_dir, 'pc_%s_'       + snapshot_name + '.ply')

    if not osp.exists(opts.result_dir):
        os.makedirs(opts.result_dir, exist_ok=True)
        LOG_INFO('"%s" directory created' % (opts.result_dir))

    data = Dataset(opts.dbname, opts.data_opts, db_root=opts.db_root, train=False)
    grids = [torch.tensor(grid, requires_grad=False).cuda() for grid in data.grids]
    eval_list = data.opts.test_idx if len(data.opts.test_idx) > 0 else data.frame_idx
    if len(eval_list) == 0:
        LOG_INFO(f'[{dbname}] No evaluation frames found. Skip.')
        return None, 0.0

    errors = np.zeros((len(eval_list), 5))
    eval_mask = np.zeros((len(eval_list),), dtype=bool)
    acc_toc = 0

    for d in range(len(eval_list)):
        fidx = eval_list[d]
        fidx_str = '%05d' % fidx

        imgs, gt, valid, raw_imgs = data.loadSample(fidx)
        net.eval()
        tic = time.time()
        imgs = [torch.Tensor(img).unsqueeze(0).cuda() for img in imgs]

        with torch.no_grad():
            invdepth_idx = net(imgs, grids, opts.valid_iters, test_mode=True)

        invdepth_idx = toNumpy(invdepth_idx[0, 0])
        invdepth     = data.indexToInvdepth(invdepth_idx)

        toc = time.time() - tic
        acc_toc += toc
        toc2 = 0.0

        if len(gt) > 0:
            eval_mask[d] = True
            errors[d, :] = data.evalError(invdepth_idx, gt, valid)

        need_vis = opts.vis or opts.save_misc or opts.save_point_cloud
        if need_vis or opts.save_result:
            tic2 = time.time()
            if need_vis:
                vis_img, inputs_rgb, pano_rgb, invdepth_rgb, err = data.makeVisImage(
                    raw_imgs, invdepth, gt, return_all=True
                )

            if opts.vis and fig is not None:
                fig.clf()
                plt.imshow(vis_img)
                plt.axis('off')
                plt.tight_layout()
                plt.draw()
                plt.pause(0.5)

            if opts.save_misc:
                writeImage(vis_img,      opts.out_misc_fmt % fidx_str)
                writeImage(inputs_rgb,   opts.out_misc_fmt.replace('misc', 'input')  % fidx_str)
                writeImage(pano_rgb,     opts.out_misc_fmt.replace('misc', 'pano')   % fidx_str)
                writeImage(invdepth_rgb, opts.out_misc_fmt.replace('misc', 'idepth') % fidx_str)
                if err is not None:
                    writeImage(err, opts.out_misc_fmt.replace('misc', 'err') % fidx_str)

            if opts.save_point_cloud:
                data.writePointCloud(pano_rgb, invdepth, opts.out_point_fmt % fidx_str)

            if opts.save_result and opts.save_float:
                _save_float_tiff(invdepth, opts.out_invdepth_fmt % fidx_str)

            toc2 += time.time() - tic2

        if eval_mask[d]:
            frame_mae = f'{errors[d, 3]:.3f}'
        else:
            frame_mae = 'N/A(no GT)'
        LOG_INFO('[%s] Process %d/%d, MAE: %s, %.3f s, misc: %.3f s' %
                 (dbname, d + 1, len(eval_list), frame_mae, toc, toc2))

    avg_time = acc_toc / len(eval_list)
    if np.any(eval_mask):
        mean_errors = errors[eval_mask].mean(axis=0)
        LOG_INFO('Result for %s: >1: %.3f, >3: %.3f, >5: %.3f, MAE: %.3f, RMS: %.3f, Avg time: %.3f' %
                 (dbname, mean_errors[0], mean_errors[1], mean_errors[2], mean_errors[3], mean_errors[4], avg_time))
        return mean_errors, avg_time

    LOG_INFO('Result for %s: No GT available. Metrics are skipped. Avg time: %.3f' %
             (dbname, avg_time))
    return None, avg_time


def main():
    if not osp.exists(opts.snapshot_path):
        sys.exit('--restore_ckpt not found: %s' % (opts.snapshot_path))
    snapshot = torch.load(opts.snapshot_path, weights_only=False)

    opts.net_opts = snapshot['net_opts']
    net = torch.nn.DataParallel(PDFOmni(opts.net_opts), device_ids=[0])
    net.load_state_dict(snapshot['net_state_dict'])

    opts.data_opts.use_rgb        = opts.net_opts.use_rgb
    opts.data_opts.num_invdepth   = opts.net_opts.num_invdepth
    opts.data_opts.num_downsample = opts.net_opts.num_downsample

    if args.dbname == 'all':
        target_datasets = ['omnithings', 'omnihouse', 'sunny', 'cloudy', 'sunset']
    else:
        target_datasets = [args.dbname]

    final_results = {}

    for db in target_datasets:
        metrics, avg_time = evaluate_dataset(opts, net, db)
        if metrics is None:
            final_results[db] = {
                '>1': None,
                '>3': None,
                '>5': None,
                'MAE': None,
                'RMS': None,
                'Time': avg_time
            }
        else:
            final_results[db] = {
                '>1': metrics[0],
                '>3': metrics[1],
                '>5': metrics[2],
                'MAE': metrics[3],
                'RMS': metrics[4],
                'Time': avg_time
            }

    if args.dbname == 'all' or len(target_datasets) > 1:
        print("\n" + "="*80)
        print(f"{'Dataset':<15} | {'>1':<8} | {'>3':<8} | {'>5':<8} | {'MAE':<8} | {'RMS':<8} | {'Time':<8}")
        print("-" * 80)

        for db in target_datasets:
            res = final_results[db]
            if res['MAE'] is None:
                print(f"{db:<15} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {res['Time']:<8.3f}")
            else:
                print(f"{db:<15} | {res['>1']:<8.3f} | {res['>3']:<8.3f} | {res['>5']:<8.3f} | {res['MAE']:<8.3f} | {res['RMS']:<8.3f} | {res['Time']:<8.3f}")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
