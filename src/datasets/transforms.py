import random
from typing import Any
import torch
from torchvision.transforms import transforms
from torchvision.transforms import functional as F


# TODO: add various transforms

def hflip(img, bbs):
    flipped_image = F.hflip(img)
    
    w, h = img.size

    # bbs of shape (N, 4)

    flipped_bbs = bbs.clone()
    flipped_bbs[:, 0] = flipped_bbs[:, 0] * -1 + 1


    # [:, [2, 1, 0, 3]] * torch.Tensor([-1, 1, -1, 1]) + torch.Tensor([1, 0, 1, 0])

    return flipped_image, flipped_bbs


class RandomHorizontalFlip:
    def __init__(self, p=.5):
        self.p = p


    def __call__(self, img, bbs):
        if random.random() < self.p:
            return hflip(img, bbs)
        return img, bbs
    
class Resize:
    def __init__(self, w=224, h=224):
        self.resize = transforms.Resize((w, h))

    def __call__(self, img, bbs):
        return self.resize(img), bbs
    

class ToTensor:
    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, img, bbs):
        return self.totensor(img), bbs
    

class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self,img, bbs):
        return self.normalize(img), bbs
    

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bbs):
        for transforms in self.transforms:
            image, bbs = transforms(image, bbs)
        
        return image, bbs
    
    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


detection_transforms = {
    'train': Compose([
        Resize(w=224, h=224),
        RandomHorizontalFlip(p=.5),
        ToTensor(),
        Normalize(),
    ]),

    'test': Compose([
        Resize(w=224, h=224),
        ToTensor(),
        Normalize(),
    ])
}
