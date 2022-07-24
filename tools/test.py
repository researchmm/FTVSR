import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmedit.apis import multi_gpu_test, set_random_seed, single_gpu_test
from mmedit.core.distributed_wrapper import DistributedDataParallelWrapper
from mmedit.datasets import build_dataloader, build_dataset
from mmedit.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='mmediting tester')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--out', help='output result pickle file')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--save-path',
        default=None,
        type=str,
        help='path to store images and if not given, will not save image')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--crf', type=int, default=0, help='test crf,0,15,25,35')
    parser.add_argument('--startIdx', type=int, default=-1, help='start idx')
    parser.add_argument('--test_frames', type=int, default=25, help='test_frames, 25, 40 ...')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def get_crf_path(crf=0):
    dcrf_dict = {
        0 : 'X4_start1',
        15: 'X4_crf15',
        25: 'X4_crf25',
        35: 'X4_crf35'
    }
    assert crf in dcrf_dict

    return dcrf_dict[crf]

def get_crf_path_vid4(crf=0):
    dcrf_dict = {
        0 : 'BD_start_1',
        15: 'BD_start_1_crf_15',
        25: 'BD_start_1_crf_25',
        35: 'BD_start_1_crf_35'
    }
    assert crf in dcrf_dict

    return dcrf_dict[crf]  

def add_path_itp(cfg):
    cfg.dataset_root = cfg.dataset_root.replace('//vzhoqiu_azsussc', os.environ['WORKDIR'])

    cfg.data.train.dataset.lq_folder = cfg.data.train.dataset.lq_folder.replace('//vzhoqiu_azsussc', os.environ['WORKDIR'])
    cfg.data.train.dataset.gt_folder = cfg.data.train.dataset.gt_folder.replace('//vzhoqiu_azsussc', os.environ['WORKDIR'])
    
    cfg.data.val.lq_folder = cfg.data.val.lq_folder.replace('//vzhoqiu_azsussc', os.environ['WORKDIR'])
    cfg.data.val.gt_folder = cfg.data.val.gt_folder.replace('//vzhoqiu_azsussc', os.environ['WORKDIR'])
    
    cfg.data.test.lq_folder = cfg.data.test.lq_folder.replace('//vzhoqiu_azsussc', os.environ['WORKDIR'])
    cfg.data.test.gt_folder = cfg.data.test.gt_folder.replace('//vzhoqiu_azsussc', os.environ['WORKDIR'])


    cfg.work_dir = osp.join(os.environ['WORKDIR'], cfg.work_dir)
    if cfg.load_from:
        cfg.load_from = osp.join(os.environ['WORKDIR'], cfg.load_from)

    return cfg


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    if cfg.itp:
        cfg = add_path_itp(cfg)
    # for crf testing
    if "REDS" in cfg.data.test.lq_folder:
        cfg.data.test.lq_folder = cfg.data.test.lq_folder.replace(get_crf_path(crf=0), get_crf_path(crf=args.crf))
        cfg.data.test.num_input_frames = args.test_frames
    elif "VID4" in cfg.data.test.lq_folder:
        cfg.data.test.lq_folder = cfg.data.test.lq_folder.replace(get_crf_path_vid4(crf=0), get_crf_path_vid4(crf=args.crf))
    else:
        raise ValueError("check test dataset: {}".format(cfg.data.test.lq_folder))

    if args.startIdx > 0:
        cfg.test_pipeline[0].start_idx = args.startIdx
        cfg.data.val.pipeline[0].start_idx = args.startIdx
        cfg.data.test.pipeline[0].start_idx = args.startIdx
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()

    # set random seeds
    if args.seed is not None:
        if rank == 0:
            print('set random seed to', args.seed)
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)

    loader_cfg = {
        **dict((k, cfg.data[k]) for k in ['workers_per_gpu'] if k in cfg.data),
        **dict(
            samples_per_gpu=1,
            drop_last=False,
            shuffle=False,
            dist=distributed),
        **cfg.data.get('test_dataloader', {})
    }

    data_loader = build_dataloader(dataset, **loader_cfg)

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    '''
    import thop
    x = torch.randn(1, 1, 3, 180, 320)
    flops, params = thop.profile(model.generator, (x,))
    flops, params = thop.clever_format([flops, params], "%.3f")
    print('flops: ', flops, "params: ", params)
    '''

    args.save_image = args.save_path is not None
    empty_cache = cfg.get('empty_cache', False)
    if not distributed:
        _ = load_checkpoint(model, args.checkpoint, map_location='cpu')
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(
            model,
            data_loader,
            save_path=args.save_path,
            save_image=args.save_image)
    else:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = DistributedDataParallelWrapper(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)

        device_id = torch.cuda.current_device()
        _ = load_checkpoint(
            model,
            args.checkpoint,
            map_location=lambda storage, loc: storage.cuda(device_id))
        outputs = multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
            save_path=args.save_path,
            save_image=args.save_image,
            empty_cache=empty_cache)

    if rank == 0:
        # print metrics
        stats = dataset.evaluate(outputs)
        psnr_msg = ''
        for stat in stats:
            print('Eval-{}: {}'.format(stat, stats[stat]))
            psnr_msg += '-{}:{:.4f}'.format(stat, stats[stat])

        # save result pickle
        if args.out:
            print('writing results to {}'.format(args.out))
            mmcv.dump(outputs, args.out)

        save_dir = cfg.work_dir
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_file = '{}_test_crf_{}.txt'.format(cfg.work_dir.split('/')[-1],args.crf)
        # save_file = 'test_crf{}_startidx{}_len{}.txt'.format(args.crf, args.startIdx, cfg.data.test.num_input_frames)
        save_file = os.path.join(save_dir, save_file)

        msg = 'crf:{}, startidx:{}, num_input_frames:{}'.format(args.crf, args.startIdx, cfg.data.test.num_input_frames) + psnr_msg + '\n'
        with open(save_file, 'a+') as f:
            f.writelines(msg)
        f.close()
        print(msg)
        print('writing file: {}'.format(save_file))


        print('*'*100)
        with open(save_file, 'r') as f:
            data = f.readlines()
        for d in data:
            print(d)


if __name__ == '__main__':
    main()
