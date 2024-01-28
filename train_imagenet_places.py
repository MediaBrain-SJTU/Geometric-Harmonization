import argparse
import time
import os
import torch
import torch.nn.functional as F
from models import SimCLR, SimCLR_imagenet
from utils import *
import numpy as np
from torch import nn
import wandb

import torch.distributed as dist
from BCL.memoboosted_LT_Dataset import Memoboosed_LT_Dataset
from SDCLR.sdclr import SDCLR, Mask

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument("--mode", type=str, default="training", help="training | test")
parser.add_argument('--data', type=str, default='', help='location of the data corpus')
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--network", type=str, default="resnet50")
parser.add_argument("--method", type=str, default="simclr")
parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
parser.add_argument('--lr_min', type=float, default=1e-6, help='minimum learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument("--epochs", type=int, default=500, help="training epochs")
parser.add_argument("--log_folder", type=str, default="./", help="log the epoch-test performance")
parser.add_argument("--checkpoint_path", type=str, default="", help="checkpoint path")
parser.add_argument('--wandb', action='store_true')
parser.add_argument("--wandb_name", type=str, default="")

parser.add_argument("--warm_up", type=int, default=0, help="warm up epoch for target initialization")
parser.add_argument('--lr_min_stage1', type=float, default=0.1)
parser.add_argument('--lr_max_stage2', type=float, default=0.1)
parser.add_argument("--GH_warm_up", type=int, default=30, help="warm up epoch for target initialization")
parser.add_argument("--target_dim", type=int, default=100, help="target dimension")
parser.add_argument('--sinkhorn_iter', default=300, type=int, help='sinkhorn iterations')
parser.add_argument('--stat_floor', default=100, type=int)
parser.add_argument('--local_rank', default=1, type=int, help='node rank for distributed training')

# BCL
parser.add_argument('--bcl', action='store_true', help='boosted contrastive learning')
parser.add_argument('--momentum_loss_beta', type=float, default=0.97)
parser.add_argument('--rand_k', type=int, default=1, help='k in randaugment')
parser.add_argument('--rand_strength', type=int, default=30, help='maximum strength in randaugment(0-30)')
# SDCLR
parser.add_argument('--prune_percent', type=float, default=0.1, help="whole prune percentage")
parser.add_argument('--random_prune_percent', type=float, default=0, help="random prune percentage")

parser.add_argument('--beta', type=float, default=0.999)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--dataset', type=str, default='imagenet', help='dataset, [imagenet, places]')
parser.add_argument('--seed', default=10, type=int)

global args
args = parser.parse_args()


def ssl_training():
    # gpus
    print("distributing")
    dist.init_process_group(backend="nccl", init_method="env://")
    print("paired")
    torch.cuda.set_device(args.local_rank)
    rank = torch.distributed.get_rank()
    setup_seed(args.seed + rank)
    world_size = torch.distributed.get_world_size()
    print("employ {} gpus in total".format(world_size))
    print("rank is {}, world size is {}".format(rank, world_size))
    assert args.batch_size % world_size == 0
    batch_size = args.batch_size // world_size

    if args.wandb and args.local_rank == 0:
        wandb.init(project='Geometric Harmonization', config=args, name=args.wandb_name, settings=wandb.Settings(start_method="fork"))

    if os.path.exists(args.log_folder) is not True:
        os.system("mkdir -p {}".format(args.log_folder))

    log = logger(path=args.log_folder, local_rank=rank, log_name="log.txt")

    # dataset
    if args.dataset == 'imagenet':
        num_class = 1000
        train_split = "split/ImageNet_LT/ImageNet_LT_train.txt"
    elif args.dataset == 'places':
        num_class = 365
        train_split = 'split/Places_LT/Places_LT_train.txt'
    if args.bcl:
        train_datasets = Memoboosed_LT_Dataset(root=args.data, txt=train_split, num_class=num_class)
    else:
        if args.network == 'resnet50':    
            train_aug, test_aug = imagenet_strong_aug()
        else:
            train_aug, test_aug = imagenet_strong_aug_norm()
        train_datasets = TwoView_LT_Dataset(root=args.data, txt=train_split, num_class=num_class, transform=train_aug)
    class_stat = np.array(train_datasets.cls_num_list)
    dataset_total_num = np.sum(class_stat)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_datasets, num_workers=8, batch_size=batch_size, 
                                               sampler=train_sampler, pin_memory=False)

    # model and loss
    if args.method == "sdclr" or args.method == "sdclr_GH":
        model = SDCLR(num_class=num_class, network=args.network).cuda()
    else:
        if args.network == 'resnet50':
            model = SimCLR_imagenet(num_class=num_class).cuda()
        else:
            model = SimCLR(num_class=num_class, network=args.network).cuda()

    process_group = torch.distributed.new_group(list(range(world_size)))
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    torch.backends.cudnn.benchmark = True

    GH_loss = GHLoss(0.05, args.sinkhorn_iter)
        
    target_weight = np.load('target/{}_128_target.npy'.format(args.target_dim))    
    target_weight = torch.tensor(target_weight).float().cuda().detach()
    logit_momentum = torch.empty((dataset_total_num, args.target_dim)).cuda()
    momentum_loss = torch.zeros(args.epochs,dataset_total_num).cuda()

    # optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.checkpoint_path, map_location="cuda")
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    # training
    loss_meter = AverageMeter("loss")
    loss_cl_meter = AverageMeter("loss_cl")
    loss_gh_meter = AverageMeter("loss_gh")
    time_meter = AverageMeter("time")
    log_interval, ckpt_interval = 50, 50
    beta = 0.9
  
    for epoch in range(start_epoch, args.epochs + 1):
        if args.warm_up > 0:
            lr_scheduler_warm(optimizer, epoch - 1, args)
            if epoch > args.warm_up:
                beta = args.beta  
        else:
            lr_scheduler(optimizer, epoch - 1, args)

        log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        train_sampler.set_epoch(epoch)

        model.train()
        loss_meter.reset(),loss_cl_meter.reset(),loss_gh_meter.reset(),time_meter.reset()
        if args.method == "sdclr" or args.method == "sdclr_GH":
            pruneMask = Mask(model)
            pruneMask.magnitudePruning(args.prune_percent - args.random_prune_percent, args.random_prune_percent)
        logit_shadow = torch.empty((dataset_total_num, args.target_dim)).cuda()
        t0 = time.time()
        torch.distributed.barrier()

        for ii, (im_x, im_y, _, index) in enumerate(train_loader):

            optimizer.zero_grad()
            batch_size = im_x.shape[0]

            if args.method != "sdclr" and args.method != "sdclr_GH":
                outs = model(im_x.cuda(non_blocking=True), im_y.cuda(non_blocking=True))

                feat = outs['z1'].detach().clone()
                logit = F.normalize(feat.mm(target_weight.t()), dim=1)
                for count in range(batch_size):
                    logit_shadow[index[count]] = logit[count].clone()

                outs['z1'] = feature_gather(outs['z1'], world_size, rank)
                outs['z2'] = feature_gather(outs['z2'], world_size, rank)

            if args.method == "simclr" or (args.method == "simclr_GH" and epoch <= args.warm_up):
                loss_cl = nt_xent(outs['z1'], t=0.2, features2=outs['z2'], average=False)
                loss = loss_cl.mean() 
                loss_cl_meter.update(loss, batch_size)
                if args.wandb and args.local_rank == 0:
                    wandb.log({"loss_cl": loss, "ep": epoch})

            elif args.method == "simclr_GH" and epoch > args.warm_up:
                loss_cl = nt_xent(outs['z1'], t=0.2, features2=outs['z2'], average=False)
                t1 = F.normalize(outs['z1'].mm(target_weight.t()), dim=1)
                t2 = F.normalize(outs['z2'].mm(target_weight.t()), dim=1)
                loss_gh = GH_loss(t1, t2, pi) 
                if epoch <= args.warm_up + args.GH_warm_up:
                    w = 1 * (epoch - args.warm_up) / args.GH_warm_up
                else:
                    w = 1
                loss = loss_cl.mean() + w * loss_gh
                loss_cl_meter.update(loss_cl.mean(), batch_size)
                loss_gh_meter.update(loss_gh, batch_size)
                if args.wandb and args.local_rank == 0:
                    wandb.log({"loss_cl": loss_cl.mean(), "loss_gh": loss_gh, "ep": epoch})

            if args.method == "focal" or (args.method == "focal_GH" and epoch <= args.warm_up):
                loss_cl = focal_nt_xent(outs['z1'], t=0.2, features2=outs['z2'], average=False)
                loss = loss_cl.mean() 
                loss_cl_meter.update(loss, batch_size)
                if args.wandb and args.local_rank == 0:
                    wandb.log({"loss_cl": loss, "ep": epoch})

            elif args.method == "focal_GH" and epoch > args.warm_up:
                loss_cl = focal_nt_xent(outs['z1'], t=0.2, features2=outs['z2'], average=False)
                t1 = F.normalize(outs['z1'].mm(target_weight.t()), dim=1)
                t2 = F.normalize(outs['z2'].mm(target_weight.t()), dim=1)
                loss_gh = GH_loss(t1, t2, pi) 
                if epoch <= args.warm_up + args.GH_warm_up:
                    w = 1 * (epoch - args.warm_up) / args.GH_warm_up
                else:
                    w = 1
                loss = loss_cl.mean() + w * loss_gh
                loss_cl_meter.update(loss_cl.mean(), batch_size)
                loss_gh_meter.update(loss_gh, batch_size)
                if args.wandb and args.local_rank == 0:
                    wandb.log({"loss_cl": loss_cl.mean(), "loss_gh": loss_gh, "ep": epoch})

            elif args.method == "sdclr" or (args.method == "sdclr_GH" and epoch <= args.warm_up):
                with torch.no_grad():
                    model.module.backbone.set_prune_flag(True)
                    features_2_noGrad = model(im_y.cuda(non_blocking=True))
                    features_2_noGrad = feature_gather(features_2_noGrad, world_size, rank).detach()
                model.module.backbone.set_prune_flag(False)
                features_1 = model(im_x.cuda(non_blocking=True))
                
                feat = features_1.detach().clone()
                logit = F.normalize(feat.mm(target_weight.t()), dim=1)
                for count in range(batch_size):
                    logit_shadow[index[count]] = logit[count].clone()

                features_1 = feature_gather(features_1, world_size, rank)
                loss_cl = nt_xent(features_1, t=0.2, features2 = features_2_noGrad, average=False)
                loss_ = loss_cl * world_size
                loss_.mean().backward()
                loss_cl_meter.update(loss_cl.mean(), batch_size)
                if args.wandb and args.local_rank == 0:
                    wandb.log({"loss_cl": loss_cl.mean(), "ep": epoch})

                features_1_no_grad = features_1.detach()
                model.module.backbone.set_prune_flag(True)
                features_2 = model(im_y.cuda(non_blocking=True))
                features_2 = feature_gather(features_2, world_size, rank)
                loss = nt_xent(features_1_no_grad, t=0.2, features2 = features_2, average=False)
                loss = (loss * world_size).mean()

            elif args.method == "sdclr_GH" and epoch > args.warm_up:
                if epoch + 1 <= args.warm_up + args.GH_warm_up:
                    w = 1 * (epoch + 1 - args.warm_up) / args.GH_warm_up
                else:
                    w = 1
                with torch.no_grad():
                    model.module.backbone.set_prune_flag(True)
                    features_2_noGrad = model(im_y.cuda(non_blocking=True))
                    features_2_noGrad = feature_gather(features_2_noGrad, world_size, rank).detach()
                model.module.backbone.set_prune_flag(False)
                features_1 = model(im_x.cuda(non_blocking=True))

                feat = features_1.detach().clone()
                logit = F.normalize(feat.mm(target_weight.t()), dim=1)
                for count in range(batch_size):
                    logit_shadow[index[count]] = logit[count].clone()

                features_1 = feature_gather(features_1, world_size, rank)
                loss_cl = nt_xent(features_1, t=0.2, features2 = features_2_noGrad, average=False)
                t1_ = F.normalize(features_1.mm(target_weight.t()), dim=1)
                t2_ = F.normalize(features_2_noGrad.mm(target_weight.t()), dim=1)
                loss_gh = GH_loss(t1_, t2_, pi)
                loss_ = (loss_cl * world_size).mean() + w * loss_gh * world_size
                loss_.backward()
                loss_cl_meter.update(loss_cl.mean(), batch_size)
                loss_gh_meter.update(loss_gh, batch_size)
                if args.wandb and args.local_rank == 0:
                    wandb.log({"loss_cl": loss_cl.mean(), "loss_gh": loss_gh, "ep": epoch})

                features_1_no_grad = features_1.detach()
                model.module.backbone.set_prune_flag(True)
                features_2 = model(im_y.cuda(non_blocking=True))
                features_2 = feature_gather(features_2, world_size, rank)
                loss_cl_ = nt_xent(features_1_no_grad, t=0.2, features2 = features_2, average=False)

                t1 = F.normalize(features_1_no_grad.mm(target_weight.t()), dim=1)
                t2 = F.normalize(features_2.mm(target_weight.t()), dim=1)
 
                loss_gh_ = GH_loss(t1, t2, pi) 
                loss = (loss_cl_ * world_size).mean() + w * loss_gh_ * world_size

            
            if args.bcl:
                for count in range(batch_size):
                    if epoch>1:
                        momentum_loss[epoch-1,index[count]] = (1.0 - args.momentum_loss_beta) * loss_cl[count].clone().detach() + args.momentum_loss_beta * momentum_loss[epoch-2,index[count]]
                    else:
                        momentum_loss[epoch-1,index[count]] = loss_cl[count].clone().detach()

            del im_x, im_y

            if args.method != "sdclr" and args.method != "sdclr_GH":
                loss = loss * world_size

            loss_meter.update(loss/world_size, batch_size)
            if args.wandb and args.local_rank == 0:
                wandb.log({"loss": loss/world_size, "ep": epoch})
            
            loss.backward()
            optimizer.step()
            time_meter.update(time.time() - t0)

            if ii % log_interval == 0:
                if args.method == "simclr" or args.method == "sdclr" or args.method == "focal":
                    log.info(f"Epoch {epoch}/{args.epochs}\tIt {ii}/{len(train_loader)}\t" +
                        f"{loss_meter}\t{time_meter}")
                elif args.method == "simclr_GH" or args.method == "sdclr_GH" or args.method == "focal_GH":
                    if epoch<=args.warm_up:
                        log.info(f"Epoch {epoch}/{args.epochs}\tIt {ii}/{len(train_loader)}\t" +
                            f"{loss_meter}\t{time_meter}")
                    else:
                        log.info(f"Epoch {epoch}/{args.epochs}\tIt {ii}/{len(train_loader)}\t" +
                            f"{loss_meter}\t{loss_cl_meter}\t{loss_gh_meter}\t{time_meter}")
            t0 = time.time()

        torch.distributed.barrier()
        torch.distributed.all_reduce(logit_shadow)
        torch.distributed.all_reduce(momentum_loss[epoch-1,:])

        if args.bcl:
            train_datasets.update_momentum_weight(momentum_loss, epoch)

        momentum_start = 10
        if epoch >= momentum_start:

            if epoch == momentum_start:
                logit_momentum = logit_shadow.clone()
            elif epoch > momentum_start:
                logit_momentum = beta * logit_momentum + (1-beta) * logit_shadow.clone()

            if (args.method == "simclr_GH" or args.method == "sdclr_GH" or args.method == "focal_GH"):
                if (epoch + 1) % 20 == 0 or epoch + 1 == args.warm_up:
                    if epoch + 1 >= args.warm_up:
                        _, target_preds = torch.max(logit_momentum, 1)
                        idxsPerClass = [np.where(np.array(target_preds.detach().cpu()) == idx)[0] for idx in range(args.target_dim)]
                        idxsNumPerClass = np.array([len(idxs) for idxs in idxsPerClass])
                        idxsNumPerClass = idxsNumPerClass + args.stat_floor # for numerical stability
                        pi = torch.tensor(idxsNumPerClass/np.sum(idxsNumPerClass)).cuda().detach()

        # save model for different epochs
        if epoch % ckpt_interval == 0 and args.local_rank == 0:
            ckpt_file = os.path.join(args.log_folder, 'ssl_{}_{}.pth'.format(args.network, epoch))
            torch.save(model.state_dict(), ckpt_file)
            log.info(f'Saved to {ckpt_file}')



if __name__ == "__main__":
    if args.mode == "training":
        ssl_training()








