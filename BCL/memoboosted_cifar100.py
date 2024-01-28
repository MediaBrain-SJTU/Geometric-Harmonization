
import torchvision.transforms as transforms
from BCL.randaug import *

def calculate_momentum_weight(momentum_loss, epoch):

    momentum_weight = ((momentum_loss-torch.mean(momentum_loss))/torch.std(momentum_loss))
    momentum_weight = ((momentum_weight/torch.max(torch.abs(momentum_weight)))/2+1/2).detach().cpu().numpy()

    return momentum_weight

class Memoboosed_Dataset(torch.utils.data.Dataset):
    r"""Returns two augmentation of each image and the image label."""

    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args

        dataset_total_num = np.sum(np.array([item for item in dataset.num_per_cls_dict.values()]))
        self.momentum_weight= np.zeros((dataset_total_num, 1))

    def update_momentum_weight(self, momentum_loss, epoch):
        momentum_weight_norm = calculate_momentum_weight(momentum_loss, epoch)
        self.momentum_weight = momentum_weight_norm

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        # img = Image.fromarray(img).convert('RGB')

        if self.args.rand_k == 1:
            # We remove the rand operation when adopt small aug 
            min_strength = 10 # training stability
            memo_boosted_aug = transforms.Compose([
                    transforms.RandomResizedCrop(32, scale=(0.1, 1.0), interpolation=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    RandAugment_prob(self.args.rand_k, min_strength + (self.args.rand_strength - min_strength)*self.momentum_weight[index], 1.0*self.momentum_weight[index]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010),
                    ),
                ])
        else:
            min_strength = 5 # training stability
            memo_boosted_aug = transforms.Compose([
                    transforms.RandomResizedCrop(32, scale=(0.1, 1.0), interpolation=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    RandAugment_prob(self.args.rand_k, min_strength + (self.args.rand_strength - min_strength)*self.momentum_weight[index]*np.random.rand(1), 1.0*self.momentum_weight[index]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010),
                    ),
                ])

        return memo_boosted_aug(img), memo_boosted_aug(img), label, index

