from torch import nn
from torch.utils.tensorboard import SummaryWriter


def write_to_tb(writer: SummaryWriter, global_index: int, net: nn.Module, scalars: dict, images: dict = {}, images_with_bbs: dict = {}):
    """Write data from training loop to tensorboard

    Args:
        writer : SummaryWriter
        global_index (int): Global Index
        net (nn.Module): Model
        lr (float): Learning rate
        scalars (dict): Dict of array of dict of scalars. E.g ->
        {
            "Loss": {
                "train": 0.9,
                "test": 0.5,
                ...
            },
            ...
        }

        images (dict): Dict of array of dict Images. Eg -> 
        {
            "FeatureMaps": {
                    "layer1.conv1": Tensor of size (C, H, W)
                    "layer2.conv2": Tensor of size (C, H, W)
                    ...
            },
            ...
        }

        images_with_bbs (dict): Dict of dicts of Images with bounding boxes. Eg ->
        {
            "Detections": {
                "0": {
                    "labels": List of length N,
                    "image": Tensor of size (C, H, W),
                    "bbs": Tensor of shape (N, 4)
                }
                ...
            },
        }
    """

    # write scalars

    for tag in scalars.keys():
        for name in scalars[tag].keys():
            value = scalars[tag][name]
            tb_tag = f"Scalars/{tag}/{name}"

            writer.add_scalar(tb_tag, value, global_index)

    # write images

    for tag in images.keys():
        for name in images[tag].keys():
            value = images[tag][name]
            tb_tag = f"Images/{tag}/{name}"

            writer.add_image(tb_tag, value, global_index)

    for tag in images_with_bbs.keys():
        for name in images_with_bbs[tag].keys():
            labels = images_with_bbs[tag][name]["labels"]
            image = images_with_bbs[tag][name]["image"]
            bbs = images_with_bbs[tag][name]["bbs"]

            tb_tag = f"Detections/{tag}/{name}"

            writer.add_image_with_boxes(
                tag=tb_tag, img_tensor=image,
                box_tensor=bbs, global_step=global_index,
                labels=labels
            )

    # parameters & gradients

    for name, parameter in net.named_parameters():
        if parameter.requires_grad and not isinstance(parameter.grad, type(None)):
            writer.add_histogram(name, parameter, global_index)
            writer.add_histogram(f"{name}.grad", parameter.grad, global_index)
