import torch
from torch.utils.data import Dataset
from .transforms import detection_transforms
from torchvision.datasets.voc import VOCDetection


def label_from_voc(vocab, annotations, width, height):
    """Convert VOC annotations to YOLOTOS format for training

    Args:
        vocab : Vocab class
        annotations : PascalVOC annotations
        width : Width of image (Used for normalization)
        height : Height of image (Used for normalization)
    """
    classes = []
    coords = []

    for annotation in annotations:
        obj_class = annotation['name']
        obj_bbox = annotation['bndbox']

        xmin = int(obj_bbox['xmin'])
        xmax = int(obj_bbox['xmax'])
        ymin = int(obj_bbox['ymin'])
        ymax = int(obj_bbox['ymax'])

        x_center = ((xmin + xmax)//2)/width  # normalized x_center
        y_center = ((ymin + ymax)//2)/height  # normalized y_center

        bb_width = (xmax - xmin)/width  # normalized bb width
        bb_height = (ymax - ymin)/height  # normalized bb height

        classes.append(obj_class)
        coords.append([x_center, y_center, bb_width, bb_height])

    classes = vocab.encode(classes)
    coords = [
        *coords,
        [0, 0, 0, 0]
    ]  #EOS is [0, 0, 0, 0]

    # it doesn't matter much what EOS is for the regression case
    # as when the model is done is dependent on the classification outputs

    return classes, coords, len(classes)


class PascalVocDataset(Dataset):
    def __init__(self, vocab, root="../data/voc", download=False, year="2007", image_set="val", shuffle=None) -> None:
        self.vocab = vocab
        self.download = download
        self.year = year
        self.image_set = image_set
        self.train = True if "train" in image_set else "test"
        self.transform = detection_transforms["train" if self.train else "test"]
        self.shuffle = shuffle or self.train

        self.dataset = VOCDetection(
            root=root, year=year,
            image_set=image_set,
            transform=None,
            download=download

        )

    def __getitem__(self, index):
        image, annotation = self.dataset[index]

        annotation = annotation["annotation"]

        w = int(annotation['size']['width'])
        h = int(annotation['size']['height'])

        annotations = annotation['object']

        classes, coords, length = label_from_voc(
            vocab=self.vocab,
            annotations=annotations,
            width=w, height=h,
        )

        classes = torch.Tensor(classes)
        coords = torch.Tensor(coords)

        image, coords = self.transform(image, coords)

        return image, classes, coords, length

    def __len__(self):
        return len(self.dataset)
