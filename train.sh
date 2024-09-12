
CUDA_NUM=5
INIT_LR=3e-4
EPOCHS=100
BATCH_SIZE=256

# List of models to train
MODELS=("EfficientNet" "ResNet" "MobileNetV2" "ViT")

# Iterate through each model and train it
for MODEL in "${MODELS[@]}"
do
    echo "Training $MODEL on CUDA $CUDA_NUM"

    # Run the Python training script with the appropriate arguments
    python main.py \
    --cuda_num $CUDA_NUM \
    --init_lr $INIT_LR \
    --epochs $EPOCHS \
    --model $MODEL \
    --batch_size $BATCH_SIZE
done

#  ./train.sh
