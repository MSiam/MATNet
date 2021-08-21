import torch
from torch.utils import data
from torchvision import transforms

import os
import sys
import time
import random
import numpy as np

from modules.MATNet import Encoder, Decoder, MATNet
from args import get_parser
from utils.utils import get_optimizer
from utils.utils import make_dir
from dataloader.dataset_utils import get_dataset_davis_youtube_ehem
from utils.utils import save_checkpoint_epoch, load_checkpoint_epoch
from utils.multigpu import setup, cleanup, find_free_port, main_process
from utils.objectives import WeightedBCE2d
from measures.jaccard import db_eval_iou_multi
from visualizer import Visualizer

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

def init_dataloaders(args, epoch_resume):
    loaders = {}

    # init dataloaders for training and validation
    for split in ['train', 'val']:
        batch_size = args.batch_size
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image_transforms = transforms.Compose([to_tensor, normalize])
        target_transforms = transforms.Compose([to_tensor])

        dataset = get_dataset_davis_youtube_ehem(
            args, split=split, image_transforms=image_transforms,
            target_transforms=target_transforms,
            augment=args.augment and split == 'train',
            inputRes=(473, 473))

        sampler = None
        if args.distributed:
            sampler = DistributedSampler(dataset)
            sampler.set_epoch(epoch_resume)
            world_size = torch.distributed.get_world_size()
            batch_size = int(batch_size / world_size)

        shuffle = True if split == 'train' and sampler is None else False
        loaders[split] = data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=args.num_workers,
                                         drop_last=True,
                                         pin_memory=True,
                                         sampler=sampler)
    return loaders

def mask_denorm(mask):
    mask_ = mask.cpu().repeat(3,1,1) * 255
    return mask_

def denorm(img):
    mean= [0.485, 0.456, 0.406]
    scale = [0.229, 0.224, 0.225]
    img = img.permute(1,2,0)
    img = img * torch.tensor(scale) + torch.tensor(mean)
    img = img.permute(2,0,1).cpu().numpy()
    img = np.asarray(img*255, np.uint8)
    return img

def trainIters(rank, world_size, args):
    print(args)

    print(f"==> Running DDP {rank}.")
    if args.distributed:
        setup(args, rank, world_size)

    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    model_dir = os.path.join(args.ckpt_path, args.model_name)
    make_dir(model_dir)

    epoch_resume = 0
    model = MATNet()

    model = model.to(rank)

    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())
    dec_opt = get_optimizer(args.optim, args.lr, decoder_params,
                            args.weight_decay)
    enc_opt = get_optimizer(args.optim_cnn, args.lr_cnn, encoder_params,
                           args.weight_decay_cnn)

    if args.distributed:
        # TODO: Fix issue in sync batchnorm not working
#        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])

#    if args.resume:
#        epoch_resume = model.resume(args)

    criterion = WeightedBCE2d(rank=rank)
    criterion.to(rank)

    loaders = init_dataloaders(args, epoch_resume)

    vis = None
    if args.vis_port != -1 and main_process(args):
        vis = Visualizer(server='http://localhost', port=args.vis_port, env=args.vis_env)

    best_iou = 0

    cur_itrs = 0

    start = time.time()
    for e in range(epoch_resume, args.max_epoch):

        print("Epoch", e)
        epoch_losses = {'train': {'total': [], 'iou': [],
                                  'mask_loss': [], 'bdry_loss': []},
                        'val': {'total': [], 'iou': [],
                                'mask_loss': [], 'bdry_loss': []}}

        for split in ['train', 'val']:
            if split == 'train':
                model.train()
            else:
                model.eval()

            for batch_idx, (image, flow, mask, bdry, negative_pixels) in\
                    enumerate(loaders[split]):
                print('Shapes: ', image.shape, ' ', flow.shape)
                cur_itrs += 1
                image, flow, mask, bdry, negative_pixels = \
                    image.to(rank), flow.to(rank), mask.to(rank), bdry.to(rank),\
                    negative_pixels.to(rank)

                if split == 'train':
                    mask_pred, p1, p2, p3, p4, p5 = model(image, flow)
#                    from pudb.remote import set_trace; set_trace(term_size=(100,24))
                    mask_loss = criterion(mask_pred, mask, negative_pixels)
                    bdry_loss = criterion(p1, bdry, negative_pixels) + \
                                criterion(p2, bdry, negative_pixels) + \
                                criterion(p3, bdry, negative_pixels) + \
                                criterion(p4, bdry, negative_pixels) + \
                                criterion(p5, bdry, negative_pixels)
                    loss = mask_loss + 0.2 * bdry_loss
                    iou = db_eval_iou_multi(mask.cpu().detach().numpy(),
                                            mask_pred.cpu().detach().numpy())

                    dec_opt.zero_grad()
                    enc_opt.zero_grad()
                    loss.backward()
                    enc_opt.step()
                    dec_opt.step()

                    if main_process(args) and vis is not None:
                        if cur_itrs % args.vis_freq == 0:
                            vis.vis_scalar('Mask Loss', cur_itrs, mask_loss.detach().cpu())
                            vis.vis_scalar('Boundary Loss', cur_itrs, bdry_loss.detach().cpu())
                            vis.vis_scalar('Total Loss', cur_itrs, loss.detach().cpu())
                            vis.vis_scalar('Train IoU', cur_itrs, iou)

                            bidx = 0
                            img_vis = denorm(image[bidx].cpu())
                            flow_vis = denorm(flow[bidx].cpu())
                            mask_vis = mask_denorm(mask[bidx])
                            mask_pred_vis = mask_denorm(mask_pred[bidx].detach())
                            neg_vis = mask_denorm(negative_pixels[bidx])

                            concat_img_mask = np.concatenate((img_vis, flow_vis, mask_vis, mask_pred_vis,
                                                              neg_vis), axis=2)
                            vis.vis_image('Image Label Pred', concat_img_mask)

                            bdry_vis = mask_denorm(bdry[bidx])
                            p1_vis = mask_denorm(p1[bidx].detach())
                            p2_vis = mask_denorm(p2[bidx].detach())
                            p3_vis = mask_denorm(p3[bidx].detach())
                            p4_vis = mask_denorm(p4[bidx].detach())
                            p5_vis = mask_denorm(p5[bidx].detach())
                            concat_img_bdry = np.concatenate((bdry_vis, p1_vis, p2_vis, p3_vis, p4_vis,
                                                              p5_vis), axis=2)
                            vis.vis_image('Boundary Label Pred', concat_img_bdry)

                else:
                    with torch.no_grad():
                        mask_pred, p1, p2, p3, p4, p5 = model(image, flow)

                        mask_loss = criterion(mask_pred, mask, negative_pixels)
                        bdry_loss = criterion(p1, bdry, negative_pixels) + \
                                    criterion(p2, bdry, negative_pixels) + \
                                    criterion(p3, bdry, negative_pixels) + \
                                    criterion(p4, bdry, negative_pixels) + \
                                    criterion(p5, bdry, negative_pixels)
                        loss = mask_loss + 0.2 * bdry_loss

                    iou = db_eval_iou_multi(mask.cpu().detach().numpy(),
                                            mask_pred.cpu().detach().numpy())

                epoch_losses[split]['total'].append(loss.data.item())
                epoch_losses[split]['mask_loss'].append(mask_loss.data.item())
                epoch_losses[split]['bdry_loss'].append(bdry_loss.data.item())
                epoch_losses[split]['iou'].append(iou)

                if (batch_idx + 1) % args.print_every == 0:
                    mt = np.mean(epoch_losses[split]['total'])
                    mmask = np.mean(epoch_losses[split]['mask_loss'])
                    mbdry = np.mean(epoch_losses[split]['bdry_loss'])
                    miou = np.mean(epoch_losses[split]['iou'])

                    te = time.time() - start
                    print('Rank: {}. Epoch: [{}/{}][{}/{}]\tTime {:.3f}s\tLoss: {:.4f}'
                          '\tMask Loss: {:.4f}\tBdry Loss: {:.4f}'
                          '\tIOU: {:.4f}'.format(rank, e, args.max_epoch, batch_idx,
                                                 len(loaders[split]), te, mt,
                                                 mmask, mbdry, miou))

                    start = time.time()

        if main_process(args):
            miou = np.mean(epoch_losses['val']['iou'])
            if args.distributed:
                dist.all_reduce(miou)
                miou /= world_size

            vis.vis_scalar('Val mIoU', e, miou)
            if miou > best_iou:
                best_iou = miou
                save_checkpoint_epoch(args, encoder, decoder,
                                      enc_opt, dec_opt, e, False)

    if args.distributed:
        cleanup()

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu_id)

    args.model_name = 'MATNet'
    #args.batch_size = 2
    args.max_epoch = 25
    args.year = '2016'

    world_size = len(args.gpu_id)
    distributed = world_size > 1
    args.distributed = distributed
    args.port = find_free_port()
    if args.distributed:
        try:
            mp.spawn(trainIters,
                     args=(world_size, args),
                     nprocs=world_size,
                     join=True)

        except KeyboardInterrupt:
            print('Interrupted')
            print('Killing processes Explicitly')
            os.system("kill $(ps ux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
    else:
       trainIters(0, 1, args)
