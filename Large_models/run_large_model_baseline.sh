#!/bin/bash
set -ex

# Large Model Baseline - To Show When ZeRO is Needed
# 512M parameter model

workdir=$HOME
master_node=ws-l6-019
worker_node=ws-l6-014

# LARGE Model configuration
TP=1
PP=1
NLAYERS=24        # 2x larger
HIDDEN=1024       # 2x larger
NUM_HEADS=16      # 2x larger

# Training configuration
GLOBAL_BATCH=16   # Slightly larger
MICRO_BATCH=2
TRAIN_ITERS=50    # Fewer iterations (takes longer)
SEQ_LENGTH=1024   # 2x longer

ZERO_STAGE=0

# Paths
BASE_PATH=$workdir/gpt
DATA_PATH=$BASE_PATH/my-gpt2_text_document
DS_CONFIG=$workdir/ds_config_large_baseline.json
HOST_FILE=$workdir/hostfile
OUTPUT_DIR=$workdir/output/LARGE_baseline_nl${NLAYERS}_hs${HIDDEN}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export MAX_JOBS=10

mkdir -p $OUTPUT_DIR
mkdir -p $BASE_PATH

if [ ! -d "$BASE_PATH/my-gpt2_text_document" ]; then
    echo "Copying dataset..."
    cp -r /l/users/omar.sayedelahl/Datasets/gpt/* $BASE_PATH/
fi

cat > $HOST_FILE << 'HOSTFILE_END'
ws-l6-019 slots=1
ws-l6-014 slots=1
HOSTFILE_END

cat > $DS_CONFIG << DSCONFIG_END
{
  "train_batch_size" : ${GLOBAL_BATCH},
  "train_micro_batch_size_per_gpu": ${MICRO_BATCH},
  "steps_per_print": 10,
  "zero_optimization": {
    "stage": ${ZERO_STAGE}
  },
  "fp16": {
    "enabled": true,
    "initial_scale_power": 12
  },
  "wall_clock_breakdown" : true
}
DSCONFIG_END

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --no-pipeline-parallel ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"

CURRENT_NODE=$(hostname)
if [ "$CURRENT_NODE" == "$master_node" ]; then
    NODE_RANK=0
    echo "Running as MASTER node (rank 0)"
else
    NODE_RANK=1
    echo "Running as WORKER node (rank 1)"
fi

echo "========================================================================"
echo "LARGE Model Baseline (512M parameters)"
echo "========================================================================"
echo "Configuration:"
echo "  Model: GPT-2 Large (${NLAYERS} layers, ${HIDDEN} hidden)"
echo "  Parameters: ~512M (vs 64M in small model)"
echo "  Global batch: ${GLOBAL_BATCH}"
echo "  Sequence length: ${SEQ_LENGTH}"
echo "  Training iterations: ${TRAIN_ITERS}"
echo "  WARNING: May run out of memory! That's the point!"
echo "========================================================================"

deepspeed --num_gpus 1 \
    --num_nodes 2 \
    --hostfile $HOST_FILE \
    --no_ssh \
    --node_rank=$NODE_RANK \
    --master_addr $master_node \
    --master_port=12345 \
    $workdir/Megatron-DeepSpeed/pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $NUM_HEADS \
    --seq-length $SEQ_LENGTH \
    --loss-scale 12 \
    --max-position-embeddings $SEQ_LENGTH \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters $TRAIN_ITERS \
    --lr 6.0e-5 \
    --min-lr 6.0e-6 \
    --lr-decay-style cosine \
    --log-interval 10 \
    --eval-iters 0 \
    --eval-interval 10000 \
    --data-path $DATA_PATH \
    --vocab-file $BASE_PATH/gpt2-vocab.json \
    --merge-file $BASE_PATH/gpt2-merges.txt \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --fp16 \
    $ds_args \
    --checkpoint-activations \
    --timing-log-level 2 \
    --vocab-size 50000 \
    --no-masked-softmax-fusion \
    | tee ${OUTPUT_DIR}/training_log_rank${NODE_RANK}.txt

echo ""
echo "Training completed!"
echo "Logs saved to: ${OUTPUT_DIR}/training_log_rank${NODE_RANK}.txt"
