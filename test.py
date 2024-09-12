import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor
from models import EfficientNet, ResNet, MobileNetV2, VisionTransformer, ViT
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# 클래스 이름 설정
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

def plot_and_save_confusion_matrix(model, test_loader, device, model_name, save_path="confusion_matrix.png"):
    model.eval()  # 모델을 평가 모드로 전환
    all_preds = []
    all_labels = []

    # 테스트 데이터로 예측 진행
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 예측값과 실제값을 리스트로 변환
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Confusion Matrix 계산
    cm = confusion_matrix(all_labels, all_preds)

    # Confusion Matrix를 백분위로 변환
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Confusion Matrix 시각화
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix for {model_name} (Percentage)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Confusion Matrix 저장
    plt.savefig(save_path)
    print(f"Confusion matrix saved as {save_path}")
    plt.close()

    # F1-score 계산
    f1 = f1_score(all_labels, all_preds, average='weighted')  # 가중 평균 F1-score 계산
    print(f"F1-score for {model_name}: {f1:.4f}")  # F1-score 출력


def load_model_and_evaluate(args):
    # Data Loading and Preprocessing
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=Compose([Resize((224, 224)), ToTensor()]),  # 이미지 크기에 맞게 조정
        # transform=Compose([Resize((28, 28)), ToTensor()]),  # 이미지 크기에 맞게 조정
    )

    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)
    device = f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu"

    # 모델 선택 및 로드
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

    # 모델 파라미터 로드
    model_path = f"Best_Model/train/{args.model.lower()}_{args.epochs}_epoch_model.pth"
    model.load_state_dict(torch.load(model_path))

    # Confusion Matrix 저장 경로 설정
    save_path = f"Best_Model/test_confusion_matrix_{args.model.lower()}.png"

    # Confusion Matrix 및 F1-score 생성 및 저장
    plot_and_save_confusion_matrix(model, test_dataloader, device, args.model, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for FashionMNIST models")
    parser.add_argument('--cuda_num', type=int, default=5, help='CUDA device number to use')
    parser.add_argument('--model', type=str, required=True,
                        choices=['EfficientNet', 'ResNet', 'MobileNetV2', 'ViT', 'ViT2'],
                        help='Model architecture to evaluate')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for evaluation')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs the model was trained for')

    args = parser.parse_args()

    load_model_and_evaluate(args)


# python test.py --model EfficientNet --epochs 100
# python test.py --model MobileNetV2 --epochs 100
# python test.py --model ViT2 --epochs 56
# python test.py --model ResNet --epochs 100
