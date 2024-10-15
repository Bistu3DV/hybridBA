#scene='mipnerf360/bicycle'
#scene='Tanks/Francis'
scene='tandt/train'
exp_name='baseline'
voxel_size=0.001
update_init_factor=16
appearance_dim=0
ratio=1
gpu=2
lod=0

# example:
#./train.sh -d ${scene} -l ${exp_name} --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio}

gs_train_iter=100

Sparse_image_folder=data/${scene} #/24_views
# SOURCE_PATH=${Sparse_image_folder}/dust3r_${N_VIEW}_views
MODEL_PATH=./output/eval/${DATASET}/${SCENE}/${N_VIEW}_views/
MODEL_PATH=/data/code0516/Scaffold-GS-hybrid-DC-barf/outputs/tandt/train/baseline/2024-08-15_09:08:22-opti-pose
# GT_POSE_PATH=${DATA_ROOT_DIR}/Tanks_colmap/${SCENE}/train_tiny #/24_views
time=$(date "+%Y-%m-%d_%H:%M:%S")

export CUDA_VISIBLE_DEVICES=${gpu}
python ./render_opti_test.py \
    -s ${Sparse_image_folder} \
    -m ${MODEL_PATH}  \
    --optim_test_pose_iter ${gs_train_iter} \
    --eval \
    --lod ${lod} \
    --voxel_size ${voxel_size} \
    --update_init_factor ${update_init_factor} \
    --appearance_dim ${appearance_dim} \
    --ratio ${ratio} \
    --iteration 30000 \
    # -m outputs/${data}/${logdir}/$time
