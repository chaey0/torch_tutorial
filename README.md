# Torch Tutorial

Pytorch implementation of EfficientNet, MobileNet, ResNet, and Vision Transformers (ViT).

## Folder Structure

```bash

├── models/                     # Model architectures
│   ├── __init__.py              # Initializes the models module
│   ├── efficientnet.py          # EfficientNet model implementation
│   ├── mobilenet.py             # MobileNet model implementation
│   ├── resnet.py                # ResNet model implementation
│   ├── ViT.py                   # Vision Transformer (ViT) model
│   ├── ViT2.py                  # Additional Vision Transformer model
│
├── main.py                      # Main script to run training and evaluation
├── test.py                      # Script to test the model on the validation set
├── train.py                     # Script to train the model
├── train.sh                     # Shell script to execute the training
├── README.md                    # Project documentation (this file)

--cuda_num: CUDA device number to use (default: 5).
--init_lr: Initial learning rate for the optimizer (default: 1e-3).
--epochs: Number of training epochs (default: 100).
--model: Model architecture to use (default: EfficientNet). Available options:
EfficientNet
ResNet
MobileNetV2
ViT (x)
ViT2
--batch_size: Batch size for training (default: 512).
