DATASET="../config/dataset.json"
MODEL="../config/model.json"
LOSS="../config/loss.json"
TRAIN="../config/train.json"

python3 ../scripts/evaluate_cms.py --dataset_config $DATASET --model_config $MODEL --loss_config $LOSS --train_config $TRAIN