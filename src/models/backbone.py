import torch
from torch import nn
from torchvision.models import vgg16, vgg19, resnet50

vgg_16_layers = {
    '0': "conv1_1",
    '1': "conv1_2",

    '5': "conv2_1",
    '7': "conv2_2",

    '10': "conv3_1",
    '12': "conv3_2",
    '14': "conv3_3",

    '17': "conv4_1",
    '19': "conv4_2",
    '21': "conv4_2",

    '24': "conv5_1",
    '26': "conv5_2",
    '28': "conv5_3",
}
vgg_19_layers = {
    **vgg_16_layers,
    '30': "conv5_4",
    '32': "conv5_5",
    '33': "conv5_6",
}


class VGG:
    def __init__(self, v="16", layers=["conv2_2", "conv5_2"]):
        super().__init__()
        assert str(v) in ["16", "19"]

        self.v = v
        self.layers = layers
        self.reference = vgg_16_layers if v == "16" else vgg_19_layers

        self.vgg_features = vgg16(
            pretrained=True
        ).features.requires_grad_(False) if v == "16" else vgg19(
            pretrained=True).features.requires_grad_(False)

    def __call__(self, x):
        features = {}

        for idx, module in self.vgg_features._modules.items():
            x = module(x)

            if str(idx) in self.reference.keys() and self.reference[str(idx)] in list(self.layers):
                features[self.reference[str(idx)]] = x

        return features


class CNNBackbone(nn.Module):
    """Pretrained CNN backbone/Encoder

    Passes the image through a pretrained CNN and get's the features

    The size of the features will be useful in determining S; the `sequence length` of the features.

    """

    def __init__(self, name: str = "vgg16"):
        super().__init__()
        """Image Encoder that makes use of pretrained CNN's

        Args:
            name (str): Must be "resnet", "vgg19" or "vgg16"
        """

        if name == "resnet":
            self.encoder = resnet50(pretrained=True).requires_grad_(False)
            self.encoder.avgpool = nn.Identity()
            self.encoder.fc = nn.Identity()
            self.embed_size = 100352  # size of output of resnet-50 before pool

        elif name == "vgg19":
            self.encoder = VGG(v="19")
            self.embed_size = 25088  # size of output of vgg-19 before pool

        else:
            self.encoder = VGG(v="16")
            self.embed_size = 25088  # size of output of vgg-19 before pool

    def forward(self, x):
        return self.encoder(x)
