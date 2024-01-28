import argparse
import os
import torch
import torch.nn.functional as F
from models import SimCLR, SimCLR_imagenet
from utils import *
import numpy as np
from torch import nn
import torch.distributed as dist
from SDCLR.sdclr import SDCLR, Mask
from collections import OrderedDict
import copy


parser = argparse.ArgumentParser(description='Testing')
parser.add_argument("--mode", type=str, default="testing", help="training | test")
parser.add_argument('--data', type=str, default='', help='location of the data corpus')
parser.add_argument("--network", type=str, default="resnet50", help="resnet50")
parser.add_argument("--method", type=str, default="simclr", help="simclr")
parser.add_argument("--log_folder", type=str, default="", help="log the epoch-test performance")
parser.add_argument("--checkpoint_path", type=str, default="", help="checkpoint path")
parser.add_argument('--local_rank', default=1, type=int, help='node rank for distributed training')
parser.add_argument('--dataset', type=str, default='imagenet', help='dataset, [imagenet, places]')
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--eval_trails', type=int, default=1)


global args
args = parser.parse_args()


def ssl_testing():
    # gpus
    print("distributing")
    dist.init_process_group(backend="nccl", init_method="env://")
    print("paired")
    torch.cuda.set_device(args.local_rank)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print("employ {} gpus in total".format(world_size))
    print("rank is {}, world size is {}".format(rank, world_size))

    if os.path.exists(args.log_folder) is not True:
        os.system("mkdir -p {}".format(args.log_folder))

    log = logger(path=args.log_folder, local_rank=rank, log_name="test_log.txt")

    if args.network == 'resnet50':    
        train_aug, test_aug = imagenet_weak_aug()
    else:
        train_aug, test_aug = imagenet_weak_aug_norm()

    # dataset
    if args.dataset == 'imagenet':
        num_class = 1000
        eval_train_datasets = LT_Dataset(root=args.data, txt='split/ImageNet_LT/ImageNet_split1_test_100shot.txt', transform=train_aug)
        eval_train_sampler_fewshot = torch.utils.data.distributed.DistributedSampler(eval_train_datasets, shuffle=True)
        eval_train_loader_fewshot = torch.utils.data.DataLoader(eval_train_datasets, num_workers=args.num_workers, batch_size=256, sampler=eval_train_sampler_fewshot)

        eval_testset = LT_Dataset(root=args.data, txt='split/ImageNet_LT/ImageNet_LT_test.txt', transform=test_aug)
        eval_test_sampler = torch.utils.data.distributed.DistributedSampler(eval_testset, shuffle=False)
        eval_test_loader = torch.utils.data.DataLoader(eval_testset, num_workers=args.num_workers, batch_size=256, sampler=eval_test_sampler)
    elif args.dataset == 'places':
        num_class = 365
        eval_train_datasets = LT_Dataset(root=args.data, txt='split/Places_LT/Places_split1_test_100shot.txt', transform=train_aug)
        eval_train_sampler_fewshot = torch.utils.data.distributed.DistributedSampler(eval_train_datasets, shuffle=True)
        eval_train_loader_fewshot = torch.utils.data.DataLoader(eval_train_datasets, num_workers=args.num_workers, batch_size=256, sampler=eval_train_sampler_fewshot)

        eval_testset = LT_Dataset(root=args.data, txt='split/Places_LT/Places_val.txt', transform=test_aug)
        eval_test_sampler = torch.utils.data.distributed.DistributedSampler(eval_testset, shuffle=False)
        eval_test_loader = torch.utils.data.DataLoader(eval_testset, num_workers=args.num_workers, batch_size=256, sampler=eval_test_sampler)
    
    if args.method == "sdclr" or args.method == "sdclr_ITA":
        model = SDCLR(num_class=num_class, network=args.network).cuda()
    else:
        if args.network == 'resnet50':
            model = SimCLR_imagenet(num_class=num_class).cuda()
        else:
            model = SimCLR(num_class=num_class, network=args.network).cuda()

    process_group = torch.distributed.new_group(list(range(world_size)))
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
    model = model.cuda()
    torch.backends.cudnn.benchmark = True

    if args.checkpoint_path != '':
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        state_dict = cvt_state_dict(state_dict, args)
        model.load_state_dict(state_dict) 

    if args.dataset=='places' or args.dataset=='imagenet':

        acc_list, majoracc_list, mediumacc_list, minoracc_list = [], [], [], []
        for i in range(args.eval_trails):

            setup_seed(10 + i)

            print('Evaluation Round-{}'.format(i))
            acc_few, perClassAcc_few = eval(eval_train_loader_fewshot, eval_test_loader, model, 0, args=args)
            acc, majoracc, mediumacc, minoracc = log_acc('fewshot',acc_few[1],perClassAcc_few,log,args=args)
            acc_list.append(acc), majoracc_list.append(majoracc), mediumacc_list.append(mediumacc), minoracc_list.append(minoracc)
        
        prefix = 'Overall'
        log.info("{}: acc is {:.02f}+-{:.02f}".format(prefix, np.mean(acc_list), np.std(acc_list)))
        log.info("{}: major acc is {:.02f}+-{:.02f}".format(prefix, np.mean(majoracc_list), np.std(majoracc_list)))
        log.info("{}: moderate acc is {:.02f}+-{:.02f}".format(prefix, np.mean(mediumacc_list), np.std(mediumacc_list)))
        log.info("{}: minor acc is {:.02f}+-{:.02f}".format(prefix, np.mean(minoracc_list), np.std(minoracc_list)))

        return


def cvt_state_dict(state_dict, args):
    # deal with adv bn
    state_dict_new = copy.deepcopy(state_dict)
    # deal with down sample layer
    name_to_del = []
    for name in np.unique(name_to_del):
        del state_dict_new[name]

    state_dict_new_noModule = OrderedDict()
    for name, item in state_dict_new.items():
        if 'module' in name:
            state_dict_new_noModule[name.replace('module.', '')] = item
    if len(state_dict_new_noModule.keys()) != 0:
        state_dict_new = state_dict_new_noModule

    return state_dict_new

def log_acc(prefix,acc_average,perClassAcc_average,log,args=None):

    if args.dataset=='places':
        accList, majorAccList, moderateAccList, minorAccList, fullVarianceList, GroupVarienceList = disjoint_summary(prefix, acc_average, perClassAcc_average, dataset='places', returnValue=True)
    elif args.dataset=='imagenet':
        accList, majorAccList, moderateAccList, minorAccList, fullVarianceList, GroupVarienceList = disjoint_summary(prefix, acc_average, perClassAcc_average, dataset='Imagenet', returnValue=True)
   
    log.info("{}: acc is {:.02f}+-{:.02f}".format(prefix, np.mean(accList), np.std(accList)))
    log.info("{}: vaiance is {:.04f}+-{:.04f}".format(prefix, np.mean(fullVarianceList), np.std(fullVarianceList)))
    log.info("{}: GroupBalancenessList is {:.04f}+-{:.04f}".format(prefix, np.mean(GroupVarienceList), np.std(GroupVarienceList)))
    log.info("{}: major acc is {:.02f}+-{:.02f}".format(prefix, np.mean(majorAccList), np.std(majorAccList)))
    log.info("{}: moderate acc is {:.02f}+-{:.02f}".format(prefix, np.mean(moderateAccList), np.std(moderateAccList)))
    log.info("{}: minor acc is {:.02f}+-{:.02f}".format(prefix, np.mean(minorAccList), np.std(minorAccList)))

    np.save(args.log_folder + '/accList.npy', accList)
    np.save(args.log_folder + '/majorAccList.npy', majorAccList)
    np.save(args.log_folder + '/moderateAccList.npy', moderateAccList)
    np.save(args.log_folder + '/minorAccList.npy', minorAccList)

    return np.mean(accList), np.mean(majorAccList), np.mean(moderateAccList), np.mean(minorAccList)


if __name__ == "__main__":
    if args.mode == "testing":
        ssl_testing()
