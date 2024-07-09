# Run with 8 GPU
DATA_PATH=$1
BART_PATH=$2

TOTAL_NUM_UPDATES="3000"
WARMUP_UPDATES="400"
LR="1e-05"
MAX_TOKENS="4096"
UPDATE_FREQ="1"
size="large"
seed="1"

# train
mkdir checkpoints
CUDA_VISIBLE_DEVICES=0 python custom_train.py --num-workers 16 ${DATA_PATH}/bins \
    --user-dir src/ \
    --seed $seed \
    --keep-best-checkpoints 3 \
    --restore-file ${BART_PATH}/model.pt \
    --max-tokens $MAX_TOKENS \
    --task text_to_table_task  --table-max-columns 16 \
    --source-lang text --target-lang data \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_ours_$size --return-relative-column-strs row_head col_head \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.05 \
    --dropout 0.2 --attention-dropout 0.2 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler inverse_sqrt --lr $LR --max-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES --warmup-init-lr '1e-07' \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --find-unused-parameters 2>&1 | tee checkpoints/log

# average checkpoints
bash scripts/eval/average_ckpt_best.sh checkpoints/
