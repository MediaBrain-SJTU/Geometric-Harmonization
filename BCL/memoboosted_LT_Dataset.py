
import torchvision.transforms as transforms
from BCL.randaug import *
from PIL import ImageFilter
import os

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def calculate_momentum_weight(momentum_loss, epoch):

    momentum_weight = ((momentum_loss[epoch-1]-torch.mean(momentum_loss[epoch-1,:]))/torch.std(momentum_loss[epoch-1,:]))
    momentum_weight = ((momentum_weight/torch.max(torch.abs(momentum_weight[:])))/2+1/2).detach().cpu().numpy()
 
    return momentum_weight
    
class Memoboosed_LT_Dataset(torch.utils.data.Dataset):

    def __init__(self, root, txt, num_class, returnPath=False):
        self.img_path = []
        self.labels = []
        self.root = root
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

        dataset_total_num = len(self.img_path)
        self.momentum_weight= np.zeros((dataset_total_num, 1))

    def __len__(self):
        return len(self.labels)

    def update_momentum_weight(self, momentum_loss, epoch):
        momentum_weight_norm = calculate_momentum_weight(momentum_loss, epoch)
        self.momentum_weight = momentum_weight_norm

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        memo_boosted_aug = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=3),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                RandAugment_prob(2, 30*self.momentum_weight[index]*np.random.rand(1), 1.0*self.momentum_weight[index]),
                transforms.ToTensor(),
            ])

        if not self.returnPath:
            return memo_boosted_aug(sample), memo_boosted_aug(sample), label, index
        else:
            return memo_boosted_aug(sample), memo_boosted_aug(sample), label, index, path.replace(self.root+'/', '')

