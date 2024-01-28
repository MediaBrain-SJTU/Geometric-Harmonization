import argparse
import os
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from datasets import ImbalanceCIFAR100
from models import SimCLR
from utils import *

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument("--mode", type=str, default="testing", help="training | test")
parser.add_argument("--gpus", type=str, default="0", help="gpu id sequence split by comma")
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument("--network", type=str, default="resnet18")
parser.add_argument("--log_folder", type=str, default="./", help="log the epoch-test performance")
parser.add_argument("--checkpoint_path", type=str, default="", help="checkpoint path")

global args
args = parser.parse_args()


def ssl_testing():
    # gpus
    gpus = list(map(lambda x: torch.device('cuda', x), [int(e) for e in args.gpus.strip().split(",")]))
    torch.cuda.set_device(gpus[0])
    torch.backends.cudnn.benchmark = True

    if os.path.exists(args.log_folder) is not True:
        os.system("mkdir -p {}".format(args.log_folder))

    log = logger(path=args.log_folder, log_name="test_log.txt")

    # dataset
    train_aug, test_aug = cifar_weak_aug()

    imb_train_dataset = ImbalanceCIFAR100("dataset/", imb_type=args.imb_type, imb_factor=args.imb_factor,
                                    rand_number=0, train=True, download=True, transform=train_aug)
    num_class = 100
    class_stat = imb_train_dataset.get_cls_num_list()

    # model and loss
    model = SimCLR(num_class=num_class, network=args.network).to(gpus[0])
    
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=gpus[0]))
    print('loading model',args.checkpoint_path)

    ### balanced classification
    eval_batch_size = 128

    eval_train_datasets = ImbalanceCIFAR100("dataset/", imb_type=args.imb_type, imb_factor=1,
                                rand_number=0, train=True, download=True, transform=train_aug)
    eval_test_dataset = torchvision.datasets.CIFAR100("dataset", train=False, download=True, transform=test_aug)

    eval_train_loader = torch.utils.data.DataLoader(eval_train_datasets, batch_size=eval_batch_size,
                                               num_workers=4, shuffle=False, pin_memory=True)
    eval_test_loader = torch.utils.data.DataLoader(eval_test_dataset, batch_size=eval_batch_size,
                                              num_workers=4, shuffle=False, pin_memory=True)

    epoch = 0
    
    acc_list, majoracc_list, mediumacc_list, minoracc_list = [], [], [], []
    
    acc, perClassAcc = eval(eval_train_loader, eval_test_loader, model, epoch, args=args)
    acc, majoracc, mediumacc, minoracc = log_acc('Accuracy',acc[1],perClassAcc,class_stat,log,args=args)
    acc_list.append(acc), majoracc_list.append(majoracc), mediumacc_list.append(mediumacc), minoracc_list.append(minoracc)

  

def log_acc(prefix,acc_average,perClassAcc_average,class_stat,log,args=None):

    accList, majorAccList, moderateAccList, minorAccList, _, _ = disjoint_summary(prefix, acc_average, perClassAcc_average, currentStatistics=class_stat, returnValue=True)
    
    log.info("{}: acc is {:.02f}+-{:.02f}".format(prefix, np.mean(accList), np.std(accList)))
    log.info("{}: major acc is {:.02f}+-{:.02f}".format(prefix, np.mean(majorAccList), np.std(majorAccList)))
    log.info("{}: moderate acc is {:.02f}+-{:.02f}".format(prefix, np.mean(moderateAccList), np.std(moderateAccList)))
    log.info("{}: minor acc is {:.02f}+-{:.02f}".format(prefix, np.mean(minorAccList), np.std(minorAccList)))

    return np.mean(accList), np.mean(majorAccList), np.mean(moderateAccList), np.mean(minorAccList)



if __name__ == "__main__":
    if args.mode == "testing":
        ssl_testing()
