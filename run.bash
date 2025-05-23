##################
# Configuration
DATASET="wikitext-103-v1"
NUM_NODES=8
DEVICES="0,1,2,3,4,5,6,7"
BATCH_SIZE=8
NUM_MICROBATCHES=1
BLOCK_SIZE=1024
N_EMBD=768
N_LAYER=12
N_HEAD=12
STAGES="3,3,3,3"    # 4 x 2 mesh
EXPNAME="$DATASET-base-pp4-dp2"
CHECKPOINT_DIR="checkpoints/$EXPNAME"
WANDB_PROJECT="$EXPNAME"

# Add a directory to the Python path
export PYTHONPATH="${PYTHONPATH}:./"

# Command string
basecmdstr="python ./examples/pp_diloco_async.py \
    --dataset $DATASET \
    --num_nodes $NUM_NODES \
    --devices $DEVICES \
    --batch_size $BATCH_SIZE \
    --num_microbatches $NUM_MICROBATCHES \
    --block_size $BLOCK_SIZE \
    --n_embd $N_EMBD \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --stages $STAGES \
    --checkpoint_dir $CHECKPOINT_DIR \
    --wandb_project $WANDB_PROJECT \
    --p_sparta 0.05 --beta1 0.99 --async_sparta_delay 10"

# Ours
cmdstr="$basecmdstr --sparta_method ema --wandb_name pp-async-Ours &"
echo $cmdstr; eval $cmdstr
wait

# DP
cmdstr="$basecmdstr --p_sparta 1.0 --async_sparta_delay 0 --wandb_name pp-async-DP &"
echo $cmdstr; eval $cmdstr
wait

# SPARTA
cmdstr="$basecmdstr --async_sparta_delay 0 --wandb_name pp-async-SPARTA &"
echo $cmdstr; eval $cmdstr
wait

# AsyncSPARTA
cmdstr="$basecmdstr --wandb_name pp-async-AsyncSPARTA &"
echo $cmdstr; eval $cmdstr
wait

# Command string
basecmdstr="python ./examples/pp_diloco_sync.py \
    --dataset $DATASET \
    --num_nodes $NUM_NODES \
    --devices $DEVICES \
    --batch_size $BATCH_SIZE \
    --num_microbatches $NUM_MICROBATCHES \
    --block_size $BLOCK_SIZE \
    --n_embd $N_EMBD \
    --n_layer $N_LAYER \
    --n_head $N_HEAD \
    --stages $STAGES \
    --checkpoint_dir $CHECKPOINT_DIR \
    --wandb_project $WANDB_PROJECT"

# FullSync
cmdstr="$basecmdstr --p_sparta 1.0 --async_sparta_delay 0 --wandb_name FullSync &"
echo $cmdstr; eval $cmdstr
wait
