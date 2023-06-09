import json
from . import voc
from . import coco
from .vocab import Vocab


def build_dataset(image_set, dataset, args):
    if dataset == "voc":
        vocab = Vocab(classes=json.load(open(f"datasets/classes/voc.json", "r")).keys())
        dataset = voc.PascalVocDataset(vocab=vocab, root=args.data_root, download=args.download_dataset,
                                       image_set=image_set, year=args.voc_year, shuffle=True)

        return dataset, vocab

    if dataset == "coco":
        vocab = Vocab(classes=json.load(open(f"datasets/classes/coco.json", "r")).keys())
        dataset = coco.CocoDataset(vocab=vocab, split=image_set)

        return dataset, vocab