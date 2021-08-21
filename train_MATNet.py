import torch
from torch.utils import data
from torchvision import transforms

import os
import sys
import time
import random
import numpy as np

from modules.MATNet import Encoder, Decoder
from args import get_parser
from utils.utils import get_optimizer
from utils.utils import make_dir, check_parallel
from dataloader.dataset_utils import get_dataset_davis_youtube_ehem
from utils.utils import save_checkpoint_epoch, load_checkpoint_epoch
from utils.objectives import WeightedBCE2d
from measures.jaccard import db_eval_iou_multi
from visualizer import Visualizer

def init_dataloaders(args):
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

        shuffle = True if split == 'train' else False
        loaders[split] = data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=args.num_workers,
                                         drop_last=True)

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

def trainIters(args):
    print(args)

    model_dir = os.path.join(args.ckpt_path, args.model_name)
    make_dir(model_dir)

    epoch_resume = 0
    if args.resume:
        encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args = \
            load_checkpoint_epoch(args.model_name, args.epoch_resume,
                                  args.use_gpu)

        epoch_resume = args.epoch_resume

        encoder = Encoder()
        decoder = Decoder()

        encoder_dict, decoder_dict = check_parallel(encoder_dict, decoder_dict)
        encoder.load_state_dict(encoder_dict)
        decoder.load_state_dict(decoder_dict)
    else:
        encoder = Encoder()
        decoder = Decoder()

    criterion = WeightedBCE2d()

    if args.use_gpu:
        encoder.cuda()
        decoder.cuda()
        criterion.cuda()

    encoder_params = list(encoder.parameters())
    decoder_params = list(decoder.parameters())
    dec_opt = get_optimizer(args.optim, args.lr, decoder_params,
                            args.weight_decay)
    enc_opt = get_optimizer(args.optim_cnn, args.lr_cnn, encoder_params,
                            args.weight_decay_cnn)

    loaders = init_dataloaders(args)

    if args.vis_port != -1:
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
                encoder.train(True)
                decoder.train(True)
            else:
                encoder.train(False)
                decoder.train(False)

            for batch_idx, (image, flow, mask, bdry, negative_pixels) in\
                    enumerate(loaders[split]):
                cur_itrs += 1
                image, flow, mask, bdry, negative_pixels = \
                    image.cuda(), flow.cuda(), mask.cuda(), bdry.cuda(),\
                    negative_pixels.cuda()

                if split == 'train':
                    r5, r4, r3, r2 = encoder(image, flow)
                    mask_pred, p1, p2, p3, p4, p5 = decoder(r5, r4, r3, r2)

                    mask_loss = criterion(mask_pred, mask, negative_pixels)
                    bdry_loss = criterion(p1, bdry, negative_pixels) + \
                                criterion(p2, bdry, negative_pixels) + \
                                criterion(p3, bdry, negative_pixels) + \
                                criterion(p4, bdry, negative_pixels) + \
                                criterion(p5, bdry, negative_pixels)
                    loss = mask_loss + 0.2 * bdry_loss
                    iou = db_eval_iou_multi(mask.cpu().detach().numpy(),
                                            mask_pred.cpu().detach().numpy())

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


                    dec_opt.zero_grad()
                    enc_opt.zero_grad()
                    loss.backward()
                    enc_opt.step()
                    dec_opt.step()
                else:
                    with torch.no_grad():
                        r5, r4, r3, r2 = encoder(image, flow)
                        mask_pred, p1, p2, p3, p4, p5 = decoder(r5, r4, r3, r2)

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
                    print('Epoch: [{}/{}][{}/{}]\tTime {:.3f}s\tLoss: {:.4f}'
                          '\tMask Loss: {:.4f}\tBdry Loss: {:.4f}'
                          '\tIOU: {:.4f}'.format(e, args.max_epoch, batch_idx,
                                                 len(loaders[split]), te, mt,
                                                 mmask, mbdry, miou))

                    start = time.time()

        miou = np.mean(epoch_losses['val']['iou'])
        vis.vis_scalar('Val mIoU', e, miou)
        if miou > best_iou:
            best_iou = miou
            save_checkpoint_epoch(args, encoder, decoder,
                                  enc_opt, dec_opt, e, False)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    args.model_name = 'MATNet'
    args.batch_size = 2
    args.max_epoch = 25
    args.year = '2016'

    gpu_id = args.gpu_id
    print('gpu_id: ', gpu_id)
    print('use_gpu: ', args.use_gpu)
    if args.use_gpu:
        torch.cuda.set_device(device=gpu_id)
        torch.cuda.manual_seed(args.seed)
    trainIters(args)
