# conda activate vode

GPU=0
IMSIZE=32

PLOT_FREQ=2
VAL_FREQ=2
# If joint
EPOCHS=100
BS=32

##############################
# JOINT WAVELET!!!!!
# 1x1x1, 3x1x1, 2x2, 4x4, 8x8, 16x16, 32x32
MODE="mrcnf"
JOINT='True'
MAXSCALES=3
NUMBLOCKS=20
BS=100
DIMS="64,64,64"
STRIDES="1,1,1,1"
EPOCHS=200
###############################


# Data
DATA='cifar10'
DATAPATH='/path/to/CIFAR10'

# Save
SAVE="/path/to/save/${DATA}/${MAXSCALES}SC${MODE}"

# Reg
KE=0.01
JF=0.01
STEER=0.0

# Model
SCALE=0
LOADCKPT=''

CONCATINPUT='True'
NOISE='True'

# OPT
LR=0.001
LRPS="0.001"
WARM=500
GAMMA=1.0
OPT='adam'

MAXGRADNORM=100.0

# Solvers
SOLVER='bosh3'
STEPSIZE=0.25
TESTSOLVER='bosh3'

# Launch your job
CUDA_VISIBLE_DEVICES=$GPU python \
  ../train_cnf_multiscale.py \
  --data $DATA --data_path $DATAPATH --im_size $IMSIZE \
  --save_path $SAVE \
  --max_scales $MAXSCALES --scale $SCALE --ckpt_to_load $LOADCKPT \
  --dims $DIMS --strides $STRIDES --num_blocks $NUMBLOCKS \
  --concat_input $CONCATINPUT --add_noise $NOISE \
  --joint $JOINT --num_epochs $EPOCHS --batch_size $BS \
  --epochs_per_scale $EPOCHSPERSCALE --batch_size_per_scale $BSPERSCALE \
  --optimizer $OPT --lr $LR --lr_per_scale $LRPS --lr_warmup_iters $WARM --lr_gamma $GAMMA --plateau_patience 10000 \
  --kinetic-energy $KE --jacobian-norm2 $JF --steer_b $STEER --max_grad_norm $MAXGRADNORM \
  --solver $SOLVER --step_size $STEPSIZE --atol 1e-3 --rtol 1e-3 \
  --test_solver $TESTSOLVER --test_atol 1e-5 --test_rtol 1e-5 \
  --plot_freq $PLOT_FREQ --val_freq $VAL_FREQ \
  --copy_scripts True \
  --seed 29 \
  # --disable_date \
