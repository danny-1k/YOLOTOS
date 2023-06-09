import torch
from torch.utils.data import Dataset
# from .transforms import detection_transforms
from torchvision.datasets.coco import CocoDetection
import matplotlib.pyplot as plt
import json

def label_from_coco(vocab, annotations, width, height, labels):
    classes = []
    coords = []
    areas = []

    for object in annotations:
        obj_class = object['category_id']
        xmin, ymin, object_width, object_height = object["bbox"]

        x_center = xmin + object_width/2  # normalized x_center
        y_center = ymin + object_height/2  # normalized y_center

        x_center = x_center / width
        y_center = y_center / height

        bb_width = object_width/width
        bb_height = object_height/height

        area = object["area"] / (width * height)

        classes.append(labels[obj_class-1])
        coords.append([x_center, y_center, bb_width, bb_height])
        areas.append(area)

    classes = vocab.encode(classes)
    coords = [
        *coords,
        [0, 0, 0, 0]
    ]  #EOS is [0, 0, 0, 0]
    areas = [*areas, 0]

    # it doesn't matter much what EOS is for the regression case
    # as when the model is done is dependent on the classification outputs

    return classes, coords, areas, len(classes)


class CocoDataset(Dataset):
    def __init__(self, vocab, split="train", shuffle=None):
        self.labels = {value: key for key, value in json.load(open("datasets/classes/coco.json", "r")).items()}
        self.vocab = vocab
        self.dataset = CocoDetection(
            root=f"../data/coco/{split}2017",
            annFile=f"../data/coco/annotations/instances_{split if split == 'train' else 'val'}2017.json",
        )

    def __getitem__(self, index):
        image, annotations = self.dataset[index]

        width, height = image.size

        classes, coords, areas, length = label_from_coco(
            vocab=self.vocab,
            annotations=annotations,
            width=width, height=height,
            labels=self.labels
        )

        return image, classes, coords, length


if __name__ == "__main__":
    import json
    from vocab import Vocab

    vocab = Vocab(classes=json.load(open(f"classes/coco.json", "r")).keys())

    dataset = CocoDataset(vocab)

    image, classes, coords, areas, length = dataset[0]
    # print(coords.shape)
    print(areas)
    # for item in coords:
    #     print(item)
    #     print()
