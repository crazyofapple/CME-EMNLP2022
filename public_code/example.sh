DEVICE=0
MODEL="bert-base-uncased"  # options: bert-base-uncased, roberta-base
TASK="QQP"  # options: SNLI, MNLI, QQP, TwitterPPDB, SWAG, HellaSWAG
IN_DOMIN_TASK="QQP"
LABEL_SMOOTHING=-1  # options: -1 (MLE), [0, 1]
MAX_SEQ_LENGTH=256
LAMB=1.0
if [ $MODEL = "bert-base-uncased" ]; then
    BATCH_SIZE=32
    LEARNING_RATE=2e-5
    WEIGHT_DECAY=0
fi

if [ $MODEL = "roberta-base" ]; then
    BATCH_SIZE=16
    LEARNING_RATE=1e-5
    WEIGHT_DECAY=0.1
fi
for seed in `seq 1 5`; do 
    CUDA_VISIBLE_DEVICES=0 python train_scaled_att_sum.py \
        --device $DEVICE \
        --model $MODEL \
        --task $TASK \
        --ckpt_path "ckpt/${IN_DOMIN_TASK}_${MODEL}_lambda_${LAMB}_${seed}.pt" \
        --output_path "output/test/${TASK}_${MODEL}_lambda_${LAMB}_${seed}.json" \
        --train_path "calibration_data/${TASK}/train.txt" \
        --dev_path "calibration_data/${TASK}/dev.txt" \
        --test_path "calibration_data/${TASK}/test.txt" \
        --epochs 3 \
        --lamb ${LAMB} \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY \
        --do_expl_selective_loss \
        --seed $seed \
        --label_smoothing $LABEL_SMOOTHING \
        --max_seq_length $MAX_SEQ_LENGTH \
        --do_train \
        --do_evaluate
    CUDA_VISIBLE_DEVICES=0 python train_scaled_att_sum.py \
        --device $DEVICE \
        --model $MODEL \
        --task $TASK \
        --ckpt_path "ckpt/${IN_DOMIN_TASK}_${MODEL}_lambda_${LAMB}_${seed}.pt" \
        --output_path "output/dev/${TASK}_${MODEL}_lambda_${LAMB}_${seed}.json" \
        --train_path "calibration_data/${TASK}/train.txt" \
        --dev_path "calibration_data/${TASK}/dev.txt" \
        --test_path "calibration_data/${TASK}/dev.txt" \
        --epochs 3 \
        --lamb ${LAMB} \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY \
        --do_expl_selective_loss \
        --seed $seed \
        --label_smoothing $LABEL_SMOOTHING \
        --max_seq_length $MAX_SEQ_LENGTH \
        --do_evaluate
done;

