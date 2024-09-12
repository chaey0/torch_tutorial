import argparse
import random, os

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Compose, RandomAutocontrast
from torchvision.transforms.v2 import RandomRotation

from models import *
from train import train, test

import torch.multiprocessing as mp

seed = 42
# Python 기본 시드 설정
random.seed(seed)
# Numpy 시드 설정
np.random.seed(seed)
# Python 기본 시드 설정
random.seed(seed)
# Numpy 시드 설정
np.random.seed(seed)
# PyTorch 시드 설정
torch.manual_seed(seed)
# CUDA 사용 시 시드 설정
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시
# CuDNN 일관성을 위해 다음 옵션 추가 (필요에 따라 선택)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(args):
    wandb.init(
        project="FashionMNIST_test",
        config={
            "learning_rate": args.init_lr,
            "architecture": args.model,
            "dataset": "FashionMNIST",
            "epochs": args.epochs,
        }
    )

    # Data Loading and Preprocessing
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=Compose([Resize((224, 224)), ToTensor()]),
        #transform=Compose([ToTensor()]),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=Compose([Resize((224, 224)), ToTensor()]),
        #transform=Compose([ToTensor()]),
    )

    train_dataloader = DataLoader(training_data, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

    device = (f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    if args.model == 'EfficientNet':
        model = EfficientNet().to(device)
    elif args.model == 'ResNet':
        model = ResNet().to(device)
    elif args.model == 'MobileNetV2':
        model = MobileNetV2().to(device)
    elif args.model == 'ViT':
        model = VisionTransformer().to(device)
    elif args.model == 'ViT2':
        model = ViT().to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=0.00005)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=5, verbose=True)

    print(f"start training with model: {args.model}")

    output_train_dir = f'Best_Model/train'
    if not os.path.exists(output_train_dir):
        os.makedirs(output_train_dir)

    output_test_dir = f'Best_Model/test'
    if not os.path.exists(output_test_dir):
        os.makedirs(output_test_dir)

    best_train_loss = 100
    best_test_loss = 100
    # Training and Testing
    for t in range(args.epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = train(train_dataloader, model, loss_fn, optimizer, t + 1, device)
        test_loss, test_acc = test(test_dataloader, model, loss_fn, t + 1, device)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_model = model

            torch.save(best_model.state_dict(), f"{output_train_dir}/{args.model.lower()}_{t + 1}_epoch_model.pth")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model = model

            torch.save(best_model.state_dict(), f"{output_test_dir}/{args.model.lower()}_{t + 1}_epoch_model.pth")

        wandb.log({
            "epoch": t + 1,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        scheduler.step(test_loss)

    print("Done!")

    # Save the model
    # torch.save(model.state_dict(), f"{args.model.lower()}_model.pth")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script for FashionMNIST models")
    parser.add_argument('--cuda_num', type=int, default=5, help='CUDA device number to use')
    parser.add_argument('--init_lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--model', type=str, default='EfficientNet',
                        choices=['EfficientNet', 'ResNet', 'MobileNetV2', 'ViT', 'ViT2'],
                        help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')

    args = parser.parse_args()

    main(args)

# python main.py --cuda_num 0 --model ResNet
# python main.py --cuda_num 1 --model EfficientNet
# python main.py --cuda_num 2 --model MobileNetV2
# python main.py --cuda_num 3 --model ViT
# python main.py --cuda_num 5 --model ViT2

