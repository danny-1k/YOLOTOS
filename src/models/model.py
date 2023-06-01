import torch
from torch import nn
from .backbone import VGG
from .attention import LuongAttention, GeneralAttention
from .matcher import build_matcher
from .criterion import CombinedDetectionLoss


class YOLOTOS(nn.Module):
    def __init__(
            self, 
            hidden_size, 
            n_tokens, 
            S=64, 
            dropout=0, 
            layers={
                "detection": {
                    "layer": "conv2_2",
                    "channels": 128,
                    "size": 56*56
                },

                "classification": { 
                    "layer": "conv5_3",
                    "channels": 512,
                    "size": 7*7
                }
            },
            backbone="vgg16"
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_tokens = n_tokens
        self.S = S
        self.dropout = dropout

        self.detection_features_layer = layers["detection"]
        self.classification_features_layer = layers["classification"]

        if "vgg" in backbone:
            self.backbone = VGG(v=backbone.replace("vgg", ""), layers=[self.detection_features_layer["layer"],
                                            self.classification_features_layer["layer"]])

        self.detection_features_project =  nn.Conv2d(
            in_channels=self.detection_features_layer["channels"],
            out_channels=S,
            kernel_size=1
        )

        self.classification_features_project = nn.Conv2d(
            in_channels=self.classification_features_layer["channels"],
            out_channels=128,
            kernel_size=1
        )

        self.box_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)
        )

        self.class_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_tokens)
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder = nn.GRU(
            input_size=self.detection_features_layer["size"] + self.classification_features_layer["size"],
            hidden_size=hidden_size,
            num_layers=1, 
            batch_first=True,
            dropout=dropout
        )

        self.decoder = nn.GRUCell(
            input_size=hidden_size, 
            hidden_size=hidden_size
        )

        self.decoder_hidden_embed = nn.Linear(hidden_size, hidden_size)

        self.attention = GeneralAttention(hidden_size=hidden_size)#LuongAttention(hidden_size=hidden_size)

    def forward(self, x, length, return_scores=False):
        batch_size = x.shape[0]

        features = self.backbone(x)

        detection_features = features[self.detection_features_layer["layer"]]
        classification_features = features[self.classification_features_layer["layer"]]

        detection_features = self.avg_pool(detection_features)
        detection_features = self.detection_features_project(detection_features)\
            .view(batch_size, self.S, -1)

        classification_features = self.max_pool(classification_features)
        classification_features = self.classification_features_project(classification_features)\
            .view(batch_size, self.S, -1)

        combined_features = torch.cat(
            (detection_features, classification_features),
            dim=2
        )

        encoder_outputs, encoder_hidden = self.encoder(combined_features)

        decoder_hidden = None

        predicted_bbs = torch.zeros(batch_size, length, 4)
        predicted_classes = torch.zeros(batch_size, length, self.n_tokens)

        if return_scores:
            attention_scores = torch.zeros(batch_size, length, encoder_outputs.shape[1])

        for t in range(length):
            if t == 0:
                decoder_hidden = self.decoder(encoder_hidden[-1],
                                              decoder_hidden)
            else:
                decoder_hidden = self.decoder(self.decoder_hidden_embed(decoder_hidden),
                                              decoder_hidden)
                
            aligned, scores = self.attention(decoder_hidden.unsqueeze(1), encoder_outputs)

            decoder_hidden = aligned.squeeze(1)

            scores = scores.squeeze(1)

            if return_scores:
                attention_scores[:, t] = scores

            predicted_bbs[:, t, :] = self.box_head(decoder_hidden)\
                .sigmoid()  # should be between 0 and 1

            predicted_classes[:, t, :] = self.class_head(decoder_hidden)  # logits

        if return_scores:
            return predicted_bbs, predicted_classes, attention_scores
        
        return predicted_bbs, predicted_classes


def build_model(args):

    S = args.S
    hidden_size = args.hidden_size
    n_tokens = args.num_classes + 2 # PAD and EOS
    dropout = args.dropout
    backbone = args.backbone
    detection_layer = args.detection_layer
    classification_layer = args.classification_layer

    layers = {
        "detection": {
        },

        "classification": {
        }
    }

    if "vgg" in backbone:
        assert VGG._has_layer(version=backbone, layer=detection_layer), f"{detection_layer} is an invalid Layer name"
        assert VGG._has_layer(version=backbone, layer=classification_layer), f"{classification_layer} is an invalid Layer name"

        net = VGG(v=backbone.lower().replace("vgg", ""), layers=[detection_layer, classification_layer])
        features = net(torch.zeros((1, 3, 224, 224)))

        detection_features_shape = features[detection_layer].shape
        classification_features_shape = features[classification_layer].shape

        layers["detection"]["layer"] = detection_layer
        layers["detection"]["channels"] = detection_features_shape[1]
        layers["detection"]["size"] = (detection_features_shape[2]//2)*(detection_features_shape[3]//2) # after pool

        layers["classification"]["layer"] = classification_layer
        layers["classification"]["channels"] = classification_features_shape[1]
        layers["classification"]["size"] = (classification_features_shape[2]//2)*(classification_features_shape[3]//2) # after pool

        net = YOLOTOS(
            hidden_size=hidden_size,
            n_tokens=n_tokens,
            S=S,
            dropout=dropout,
            layers=layers,
            backbone=backbone.lower()
        )

        matcher = build_matcher(args)

        criterion = CombinedDetectionLoss(num_tokens=n_tokens, matcher=matcher, use_matcher=args.use_matcher)
        
        return net, criterion


if __name__ == "__main__":

    image = torch.rand((1, 3, 224, 224))

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden_size", default=256)
    parser.add_argument("--num_classes", default=25)
    parser.add_argument("--S", default=64)
    parser.add_argument("--dropout", default=0)
    parser.add_argument("--backbone", default="vgg16")
    parser.add_argument("--detection_layer", default="conv2_2")
    parser.add_argument("--classification_layer", default="conv5_3")


    args = parser.parse_args()

    model = build_model(args)

    bbs, classes, scores = model(image, 10, return_scores=True)

    print(bbs.shape, classes.shape, scores.shape)