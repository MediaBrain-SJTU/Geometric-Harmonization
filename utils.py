import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import os
import torch.optim as optim
import re
from PIL import Image, ImageFilter
import random
from torch.utils.data import Dataset

class GHLoss(nn.Module):
    def __init__(self, l, sinkhorn_iterations):
        super().__init__()
        self.l = l
        self.sinkhorn_iterations = sinkhorn_iterations

    def loss(self, x, y, stat):
        gamma = 0.1
        q_x = sinkhorn(x, self.l, self.sinkhorn_iterations, stat)
        q_y = sinkhorn(y, self.l, self.sinkhorn_iterations, stat)
        return  -torch.mean(torch.sum(q_x * F.log_softmax(y/gamma, dim=1), dim=1))-\
                torch.mean(torch.sum(q_y * F.log_softmax(x/gamma, dim=1), dim=1))

    def forward(self, c1, c2, stat):
        loss = self.loss(c1, c2, stat)

        return loss

@torch.no_grad()
def sinkhorn(out, l, sinkhorn_iterations, stat=None):

    Q = torch.exp(out / l).t()
    
    N = Q.shape[1] 
    K = Q.shape[0] 

    r = (torch.ones((K, 1)) / K).cuda()
    c = (torch.ones((N, 1)) / N).cuda()
    if stat is None:
        inv_K = 1. / K    
    else:
        inv_K = stat.clone().detach().reshape(K,1).float()
        inv_K = inv_K/torch.sum(inv_K)
    inv_N = 1. / N

    for _ in range(sinkhorn_iterations):
        r = inv_K / (Q @ c)    
        c = inv_N / (Q.T @ r)  

    Q = r * Q * c.t()
    Q = Q.t()

    Q *= N 

    return Q


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def nt_xent(x, t=0.5, features2=None, average=True):

    if features2 is None:
        out = F.normalize(x, dim=-1)
        d = out.size()
        batch_size = d[0] // 2
        out = out.view(batch_size, 2, -1).contiguous()
        out_1 = out[:, 0]
        out_2 = out[:, 1]
    else:
        batch_size = x.shape[0]
        out_1 = F.normalize(x, dim=-1)
        out_2 = F.normalize(features2, dim=-1)

    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / t)

    # mask = get_negative_mask(batch_size).cuda(non_blocking=True)
    mask = get_negative_mask(batch_size).cuda()

    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / t)
    pos = torch.cat([pos, pos], dim=0)

    # estimator g()
    Ng = neg.sum(dim=-1)
    
    # contrastive loss
    loss = (- torch.log(pos / (pos + Ng)))
    loss_reshape = loss.view(2, batch_size).mean(0)

    if average:
        # return loss.mean()
        return loss_reshape.mean()
    else: 
        return loss_reshape
    
def focal_nt_xent(x, t=0.5, gamma=2, features2=None, average=True):

    if features2 is None:
        out = F.normalize(x, dim=-1)
        d = out.size()
        batch_size = d[0] // 2
        out = out.view(batch_size, 2, -1).contiguous()
        out_1 = out[:, 0]
        out_2 = out[:, 1]
    else:
        batch_size = x.shape[0]
        out_1 = F.normalize(x, dim=-1)
        out_2 = F.normalize(features2, dim=-1)

    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / t)

    mask = get_negative_mask(batch_size).cuda()
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / t)
    pos = torch.cat([pos, pos], dim=0)

    # estimator g()
    Ng = neg.sum(dim=-1)
    
    # contrastive loss
    p = pos / (pos + Ng)
    loss = (-(1-p)**gamma*torch.log(p))
    loss_reshape = loss.view(2, batch_size).mean(0)

    if average:
        return loss.mean()
    else: 
        # return loss_reshape, (1-p)**gamma
        return loss_reshape

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def cifar_strong_aug():
    train_aug = transforms.Compose([
        torchvision.transforms.RandomResizedCrop(32, scale=(0.1, 1)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomApply([
            torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    test_aug = transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.CenterCrop(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    return train_aug, test_aug

def cifar_weak_aug():
    train_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return train_aug, test_aug

def imagenet_strong_aug():

    train_aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
    ])

    test_aug = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ])

    return train_aug, test_aug

def imagenet_strong_aug_norm():

    train_aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_aug = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    return train_aug, test_aug


def imagenet_weak_aug():

    train_aug = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    test_aug = transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          ])

    return train_aug, test_aug

def imagenet_weak_aug_norm():

    train_aug = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    test_aug = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    return train_aug, test_aug


class TwoView_LT_Dataset(Dataset):

  def __init__(self, root, txt, num_class, transform=None, returnPath=False):
    self.img_path = []
    self.labels = []
    self.root = root
    self.transform = transform
    self.returnPath = returnPath
    self.txt = txt
    self.num_class = num_class

    with open(txt) as f:
        for line in f:
            self.img_path.append(os.path.join(root, line.split()[0]))
            self.labels.append(int(line.split()[1]))

    self.class_data=[[] for i in range(self.num_class)]
    for i in range(len(self.labels)):
        y=self.labels[i]
        self.class_data[y].append(i)

    self.cls_num_list=[len(self.class_data[i]) for i in range(self.num_class)]

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):

    path = self.img_path[index]
    label = self.labels[index]

    with open(path, 'rb') as f:
      sample = Image.open(f).convert('RGB')

    if not self.returnPath:
      return self.transform(sample), self.transform(sample), label, index
    else:
      return self.transform(sample), self.transform(sample), label, index, path.replace(self.root+'/', '')


class TwoViewAugDataset_index(torch.utils.data.Dataset):
    r"""Returns two augmentation of each image and the image label."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), self.transform(image), label, index

    def __len__(self):
        return len(self.dataset)
    
class Dataset_index(torch.utils.data.Dataset):
    r"""Returns two augmentation of each image and the image label."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label, index

    def __len__(self):
        return len(self.dataset)


class LT_Dataset(Dataset):

  def __init__(self, root, txt, transform=None, returnPath=False):
    self.img_path = []
    self.labels = []
    self.root = root
    self.transform = transform
    self.returnPath = returnPath
    self.txt = txt

    with open(txt) as f:
      for line in f:
        self.img_path.append(os.path.join(root, line.split()[0]))
        self.labels.append(int(line.split()[1]))

    self.targets = self.labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):

    path = self.img_path[index]
    label = self.labels[index]

    with open(path, 'rb') as f:
      sample = Image.open(f).convert('RGB')

    if self.transform is not None:
      sample = self.transform(sample)

    if not self.returnPath:
      return sample, label
    else:
      return sample, label, index, path.replace(self.root+'/', '')


def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))

    return lr


def eval_knn(x_train, y_train, x_test, y_test, k=5):
    """ k-nearest neighbors classifier accuracy """
    d = torch.cdist(x_test, x_train)
    topk = torch.topk(d, k=k, dim=1, largest=False)
    labels = y_train[topk.indices]
    pred = torch.empty_like(y_test)
    for i in range(len(labels)):
        x = labels[i].unique(return_counts=True)
        pred[i] = x[0][x[1].argmax()]

    acc = (pred == y_test).float().mean().cpu().item()
    del d, topk, labels, pred
    return acc

def lr_scheduler_warm(optimizer, epoch, args):
        epoch = epoch + 1
        lr_max, lr_min = args.lr, args.lr_min
        lr_min_stage1, lr_max_stage2 = args.lr_min_stage1, args.lr_max_stage2
        if epoch < 10:
            lr = lr_max * epoch / 10
        elif epoch < args.warm_up:
            lr =  lr_min_stage1 + (lr_max - lr_min_stage1) * 0.5 * (1 + np.cos((epoch - 10) / (args.warm_up - 10) * np.pi))
        else:
            lr =  lr_min + (lr_max_stage2 - lr_min) * 0.5 * (1 + np.cos((epoch - args.warm_up) / (args.epochs - args.warm_up) * np.pi))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def lr_scheduler(optimizer, epoch, args):
        epoch = epoch + 1
        lr_max = args.lr
        lr_min = args.lr_min
        if epoch < 10:
            lr = lr_max * epoch / 10
        else:
            lr =  lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((epoch - 10) / (args.epochs - 10) * np.pi))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def eval_sgd(x_train, y_train, x_test, y_test, topk=[1, 5], epoch=500, batch_size=1000):
    
    """ linear classifier accuracy (sgd) """
    lr_start, lr_end = 1e-2, 1e-6
    gamma = (lr_end / lr_start) ** (1 / epoch)
    output_size = x_train.shape[1]
    num_class = y_train.max().item() + 1
    clf = nn.Linear(output_size, num_class)

    clf.cuda()

    clf.train()
    optimizer = optim.Adam(clf.parameters(), lr=lr_start, weight_decay=5e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epoch):
    
        perm = torch.randperm(len(x_train))
        n_batch = int(np.ceil(len(x_train)*1.0/batch_size))
        for ii in range(n_batch):
            optimizer.zero_grad()
            mask = perm[ii*batch_size:(ii+1)*batch_size]
            criterion(clf(x_train[mask]), y_train[mask]).backward()
            optimizer.step()
        
        scheduler.step()

    clf.eval()
    with torch.no_grad():
        y_pred = clf(x_test)
    pred_top = y_pred.topk(max(topk), 1, largest=True, sorted=True).indices
    acc = {
        t: (pred_top[:, :t] == y_test[..., None]).float().sum(1).mean().cpu().item()
        for t in topk
    }

    del clf
    return acc

def eval_sgd_per_class(x_train, y_train, x_test, y_test, topk=[1, 5], epoch=500, batch_size=1000):
    
    num_class = y_train.max().item() + 1

    perClassAccRight = [0 for _ in range(num_class)]
    perClassAccWhole = [0 for _ in range(num_class)]
    perClassAcc = [0 for _ in range(num_class)]
    
    """ linear classifier accuracy (sgd) """
    lr_start, lr_end = 1e-2, 1e-6
    gamma = (lr_end / lr_start) ** (1 / epoch)
    output_size = x_train.shape[1]
    clf = nn.Linear(output_size, num_class)

    clf.cuda()

    clf.train()
    optimizer = optim.Adam(clf.parameters(), lr=lr_start, weight_decay=5e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epoch):
        # add
        perm = torch.randperm(len(x_train))
        n_batch = int(np.ceil(len(x_train)*1.0/batch_size))
        for ii in range(n_batch):
            optimizer.zero_grad()
            mask = perm[ii*batch_size:(ii+1)*batch_size]
            criterion(clf(x_train[mask]), y_train[mask]).backward()
            optimizer.step()
        
        scheduler.step()

    clf.eval()
    with torch.no_grad():
        y_pred = clf(x_test)
    pred_top = y_pred.topk(max(topk), 1, largest=True, sorted=True).indices
    acc = {
        t: (pred_top[:, :t] == y_test[..., None]).float().sum(1).mean().cpu().item()
        for t in topk
    }

    y_pred = y_pred.max(1, keepdim=True)[1]

    for i in range(num_class):
        perClassAccRight[i] = y_pred[y_test == i].eq(y_test[y_test == i].view_as(y_pred[y_test == i])).sum().item()
        perClassAccWhole[i] = len(y_test[y_test == i])

    for i in range(num_class):
        perClassAcc[i] = perClassAccRight[i] / perClassAccWhole[i] * 100

    del clf
    return acc, perClassAcc

def eval(train_loader, test_loader, model, epoch, class_stat=None, args=None):
    
    model.eval()
    
    if args.method == 'sdclr' or args.method == 'sdclr_GH':
        projector = model.backbone.fc
        model.backbone.fc = nn.Identity()
    else:
        model_ = model
        model = model.backbone

    with torch.no_grad():

        model.eval()
        x_train = []
        y_train = []

        x_test = []
        y_test = []

        for i, (inputs, labels) in enumerate(train_loader):
        
            inputs = inputs.cuda()
            features = model(inputs)

            x_train.append(features.detach())
            y_train.append(labels.detach())

        for i, (inputs, labels) in enumerate(test_loader):
            
            inputs = inputs.cuda()
            features = model(inputs)

            x_test.append(features.detach())
            y_test.append(labels.detach())

        x_train = torch.cat(x_train, dim=0)
        y_train = torch.cat(y_train, dim=0).cuda()

        x_test = torch.cat(x_test, dim=0)
        y_test = torch.cat(y_test, dim=0).cuda()

    acc, perClassAcc = eval_sgd_per_class(x_train, y_train, x_test, y_test)

    if args.method == 'sdclr' or args.method == 'sdclr_GH':
        model.backbone.fc = projector
    else:
        model = model_

    return acc, perClassAcc

class AverageMeter(object):
    r"""
    Computes and stores the average and current value.
    Adapted from
    https://github.com/pytorch/examples/blob/ec10eee2d55379f0b9c87f4b36fcf8d0723f45fc/imagenet/main.py#L359-L380
    """
    def __init__(self, name=None, fmt='.6f'):
        fmtstr = f'{{val:{fmt}}} ({{avg:{fmt}}})'
        if name is not None:
            fmtstr = name + ' ' + fmtstr
        self.fmtstr = fmtstr
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        avg = self.sum / self.count
        if isinstance(avg, torch.Tensor):
            avg = avg.item()
        return avg

    def __str__(self):
        val = self.val
        if isinstance(val, torch.Tensor):
            val = val.item()
        return self.fmtstr.format(val=val, avg=self.avg)

class logger(object):
    def __init__(self, path, log_name="log.txt", local_rank=0):
        self.path = path
        self.local_rank = local_rank
        self.log_name = log_name

    def info(self, msg):
        if self.local_rank == 0:
            print(msg)
            with open(os.path.join(self.path, self.log_name), 'a') as f:
                f.write(msg + "\n")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def feature_gather(features, world_size, rank):

    features_list = [torch.zeros_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(features_list, features)
    features_list[rank] = features
    features = torch.cat(features_list)

    return features

def getStatisticsFromTxt(txtName, num_class=1000):
    statistics = [0 for _ in range(num_class)]
    with open(txtName, 'r') as f:
        lines = f.readlines()
    for line in lines:
        s = re.search(r" ([0-9]+)$", line)
        if s is not None:
            statistics[int(s[1])] += 1
    return statistics


def disjoint_summary(prefix, bestAcc, classWiseAcc, currentStatistics=None, dataset='cifar10',
                      noReturnAvg=False, returnValue=False, group=3, noGroup=False):

    accList = []
    fullVarianceList = []
    GroupVarienceList = []
    majorAccList = []
    moderateAccList = []
    minorAccList = []

    # get major moderate minor class accuracy
    if currentStatistics is None:
        if dataset == 'Imagenet':
            # currentStatistics = np.array(getStatisticsFromTxt('split/ImageNet_LT/imageNet_LT_exp_train.txt'))
            currentStatistics = np.array(getStatisticsFromTxt('split/ImageNet_LT/ImageNet_LT_train.txt'))
        elif dataset == 'places':
            currentStatistics = np.array(getStatisticsFromTxt('split/Places_LT/Places_LT_train.txt', num_class=365))
        elif dataset == 'inat':
            currentStatistics = np.array(getStatisticsFromTxt('split/iNaturalist18/iNaturalist18_train.txt', num_class=8142))
        elif dataset == 'marine':
            currentStatistics = np.array(getStatisticsFromTxt('split/marine/mt_train.txt', num_class=60))
        else:
            assert False

    sortIdx = np.argsort(currentStatistics)
    idxsMajor = sortIdx[len(currentStatistics) // 3 * 2:]
    idxsModerate = sortIdx[len(currentStatistics) // 3 * 1: len(currentStatistics) // 3 * 2]
    idxsMinor = sortIdx[: len(currentStatistics) // 3 * 1]

    classWiseAcc = np.array(classWiseAcc)
    bestAcc = np.mean(classWiseAcc)
    majorAcc = np.mean(classWiseAcc[idxsMajor])
    moderateAcc = np.mean(classWiseAcc[idxsModerate])
    minorAcc = np.mean(classWiseAcc[idxsMinor])

    if dataset == "Imagenet" or dataset == "places" or dataset == "inat":
        idxsMany = np.nonzero(currentStatistics > 100)[0]
        idxsMedium = np.nonzero((100 >= currentStatistics) & (currentStatistics >= 20))[0]
        idxsFew = np.nonzero(currentStatistics < 20)[0]
        majorAcc = np.mean(classWiseAcc[idxsMany])
        moderateAcc = np.mean(classWiseAcc[idxsMedium])
        minorAcc = np.mean(classWiseAcc[idxsFew])

    accList = classWiseAcc
    majorAccList = classWiseAcc[idxsMajor]
    moderateAccList = classWiseAcc[idxsModerate]
    minorAccList = classWiseAcc[idxsMinor]

    fullVarianceList.append(np.std(classWiseAcc / 100))
    GroupVarienceList.append(np.std(np.array([majorAcc, moderateAcc, minorAcc]) / 100))

    if group > 3:
        assert len(classWiseAcc) % group == 0
        group_idx_list = [sortIdx[len(currentStatistics) // group * cnt: len(currentStatistics) // group * (cnt + 1)] \
                            for cnt in range(0, group)]
        group_accs = [np.mean(classWiseAcc[group_idx_list[cnt]]) for cnt in range(0, group)]
        outputStr = "{}: group accs are".format(prefix)
        for acc in group_accs:
            outputStr += " {:.02f}".format(acc)
        print(outputStr)

    if returnValue:
        return accList, majorAccList, moderateAccList, minorAccList, fullVarianceList, GroupVarienceList
    else:
        if noReturnAvg:
            outputStr = "{}: accs are".format(prefix)
            for acc in accList:
                outputStr += " {:.02f}".format(acc)
            print(outputStr)
            if not noGroup:
                outputStr = "{}: majorAccs are".format(prefix)
                for acc in majorAccList:
                    outputStr += " {:.02f}".format(acc)
                print(outputStr)
                outputStr = "{}: moderateAccs are".format(prefix)
                for acc in moderateAccList:
                    outputStr += " {:.02f}".format(acc)
                print(outputStr)
                outputStr = "{}: minorAccs are".format(prefix)
                for acc in minorAccList:
                    outputStr += " {:.02f}".format(acc)
            print(outputStr)
        else:
            print("{}: acc is {:.02f}+-{:.02f}".format(prefix, np.mean(accList), np.std(accList)))
            if not noGroup:
                print("{}: vaiance is {:.04f}+-{:.04f}".format(prefix, np.mean(fullVarianceList), np.std(fullVarianceList)))
                print("{}: GroupBalancenessList is {:.04f}+-{:.04f}".format(prefix, np.mean(GroupVarienceList), np.std(GroupVarienceList)))
                print("{}: major acc is {:.02f}+-{:.02f}".format(prefix, np.mean(majorAccList), np.std(majorAccList)))
                print("{}: moderate acc is {:.02f}+-{:.02f}".format(prefix, np.mean(moderateAccList), np.std(moderateAccList)))
                print("{}: minor acc is {:.02f}+-{:.02f}".format(prefix, np.mean(minorAccList), np.std(minorAccList)))