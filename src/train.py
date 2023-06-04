import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from models import build_model
from datasets import build_dataset
from utils.data import dataloader_collate_fn
from utils.model import clean_prediction
from utils.box_ops import box_cxcywh_to_xyxy
from utils.tensorboard import write_to_tb
from utils.misc import denormalize
from tqdm import tqdm


def train_one_epoch(net, criterion, data, optimizer, device, iteration, writer, vocab, visualize_attention):
    net.train()
    criterion.train()

    net.to(device)
    criterion.to(device)

    average_loss = 0
    average_bb_loss = 0
    average_giou_loss = 0
    average_class_loss = 0


    for image, target_classes,  target_bbs, lengths in tqdm(data):
        image = image.to(device)
        target_classes = target_classes.to(device).long()
        target_bbs = target_bbs.to(device)

        optimizer.zero_grad()
        max_len = max(lengths)

        if visualize_attention:
            predicted_bbs, predicted_classes, attention_scores = net(image, max_len, return_scores=visualize_attention)
        else:
            predicted_bbs, predicted_classes = net(image, max_len, return_scores=visualize_attention)

        loss, loss_dict = criterion(
            {
                "bbs": predicted_bbs,
                "classes": predicted_classes
            }, 

            {
                "bbs": target_bbs,
                "classes": target_classes
            }
        )

        loss.backward()

        optimizer.step()

        
        one_example_class_prediction = predicted_classes[0].argmax(-1)
        one_example_bb_prediction = predicted_bbs[0]

        one_example_class_target = target_classes[0]
        one_example_bb_target = target_bbs[0]

        one_example_class_prediction, one_example_bb_prediction = clean_prediction(
            classes=one_example_class_prediction,
            bbs=one_example_bb_prediction
        )

        one_example_class_target, one_example_bb_target = clean_prediction(
            classes=one_example_class_target,
            bbs=one_example_bb_target
        )

        one_example_bb_prediction = box_cxcywh_to_xyxy(one_example_bb_prediction) * 244
        one_example_bb_target = box_cxcywh_to_xyxy(one_example_bb_target) * 244

        one_example_labels_predicted_decoded = vocab.decode(one_example_class_prediction.tolist())
        one_example_labels_target_decoded = vocab.decode(one_example_class_target.tolist())

        iteration += 1

        average_loss = .6*average_loss + .4*loss.item()
        average_bb_loss = .6*average_bb_loss + .4*loss_dict["bb_loss"]
        average_giou_loss = .6*average_giou_loss + .4*loss_dict["giou_loss"]
        average_class_loss = .6*average_class_loss + .4*loss_dict["class_loss"]


    write_to_tb(
        writer=writer,
        global_index=iteration,
        net=net,
        scalars={
            "Loss/train": {
                "combined": average_loss,
                "bb": average_bb_loss,
                "giou": average_giou_loss,
                "class": average_class_loss
            }
        },
        images_with_bbs={
            "train": {
                "predicted": {
                    "labels": one_example_labels_predicted_decoded,
                    "image": denormalize(image[0]),
                    "bbs": one_example_bb_prediction
                },

                "target": {
                    "labels": one_example_labels_target_decoded,
                    "image": denormalize(image[0]),
                    "bbs": one_example_bb_target
                },

            }
        },

        images={
            "Attention/train": {
                "0": attention_scores[0].cpu().unsqueeze(0)
            }
        } if visualize_attention else {}
    )


    return iteration, average_loss

@torch.no_grad()
def eval_model(net, criterion, data, device, iteration, writer):
    net.eval()
    criterion.eval()

    net.to(device)
    criterion.to(device)

    average_loss = 0
    average_bb_loss = 0
    average_giou_loss = 0
    average_class_loss = 0


    for image, target_classes,  target_bbs, lengths in tqdm(data):
        image = image.to(device)
        target_classes = target_classes.to(device).long()
        target_bbs = target_bbs.to(device)

        max_len = max(lengths)

        predicted_bbs, predicted_classes = net(image, max_len, return_scores=False)

        loss, loss_dict = criterion(
            {
                "bbs": predicted_bbs,
                "classes": predicted_classes
            }, 

            {
                "bbs": target_bbs,
                "classes": target_classes
            }
        )


        average_loss = .6*average_loss + .4*loss.item()
        average_bb_loss = .6*average_bb_loss + .4*loss_dict["bb_loss"]
        average_giou_loss = .6*average_giou_loss + .4*loss_dict["giou_loss"]
        average_class_loss = .6*average_class_loss + .4*loss_dict["class_loss"]


    write_to_tb(
        writer=writer,
        global_index=iteration,
        net=net,
        scalars={
            "Loss/test": {
                "combined": average_loss,
                "bb": average_bb_loss,
                "giou": average_giou_loss,
                "class": average_class_loss
            }
        },
    )

    return average_loss


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dataset == "voc":
        args.num_classes = 20

    device = args.device

    net, criterion = build_model(args)

    net.backbone.to(device)

    net.to(device)
    criterion.to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # create necessary folders

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if not os.path.exists(os.path.join(args.checkpoint_dir, args.run_name)):
        os.makedirs(os.path.join(args.checkpoint_dir, args.run_name))


    writer = SummaryWriter(log_dir=f"{args.logdir}/{args.run_name}")


    train, vocab = build_dataset(
        image_set="val" if args.no_eval else "train",
        dataset=args.dataset, 
        args=args
    )

    train = DataLoader(
        train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=dataloader_collate_fn
    )

    if not args.no_eval:

        test, _ = build_dataset(
            image_set="val",
            dataset=args.dataset,
            args=args
        )

        test = DataLoader(
            test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=dataloader_collate_fn
        )


    number_of_train_iters = 0

    print("Started Training.")

    for epoch in range(args.epochs):

        number_of_train_iters, average_train_loss = train_one_epoch(
            net=net,
            criterion=criterion,
            data=train,
            optimizer=optimizer,
            device=device,
            iteration=number_of_train_iters,
            writer=writer,
            vocab=vocab,
            visualize_attention=args.visualize_attention
        )


        lr_scheduler.step()


        for checkpoint_path in [f"{args.checkpoint_dir}/{args.run_name}/checkpoint.pth", f"{args.checkpoint_dir}/{args.run_name}/checkpoint{epoch: 04}.pth"]:
            torch.save({
                "net": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "loss": average_train_loss,
            }, checkpoint_path)


        if not args.no_eval:
            average_test_loss = eval_model(
                net=net,
                criterion=criterion,
                data=test,
                device=device,
                iteration=number_of_train_iters,
                writer=writer,
            )



if __name__ == "__main__":
    from argparse import ArgumentParser
    from utils.misc import get_hostname_and_time_string

    unique_string = get_hostname_and_time_string()

    parser = ArgumentParser()

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument("--lr_drop", default=100, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument("--visualize_attention", action="store_true")

    # Network
    parser.add_argument('--backbone', default='vgg16', type=str,
                        help="Encoder. `vgg16`, `vgg19`")

    parser.add_argument('--hidden_size', default=256, type=int,
                        help="Hidden size of decoder GRU")
    
    parser.add_argument('--dropout', default=0,
                        type=float, help="Dropout")
    

    parser.add_argument("--S", default=64, type=int)
    parser.add_argument("--detection_layer", default="conv2_2", type=str)
    parser.add_argument("--classification_layer", default="conv5_3", type=str)

    # Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # Loss
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)

    # Misc
    parser.add_argument('--dataset', default='voc', type=str)
    parser.add_argument('--voc_year', default='2007', type=str)
    parser.add_argument("--data_root", default="../data/voc", type=str)
    parser.add_argument("--download_dataset", action="store_true")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--seed', default=3417, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument("--logdir", default="../logs",
                        type=str, help="Logdir for tensorboard")
    
    parser.add_argument("--run_name", default=unique_string,
                         type=str, help="Unique string for saving checkpoints and tensorboard")
    
    parser.add_argument("--checkpoint_dir", default="../checkpoints/")

    parser.add_argument("--use_matcher", action="store_true")

    parser.add_argument("--no_eval", action="store_true")

    args = parser.parse_args()


    run(args)
