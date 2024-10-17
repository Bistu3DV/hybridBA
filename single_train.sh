scene='tandt/train'
exp_name='baseline'

voxel_size=0.001
update_init_factor=16
appearance_dim=0
ratio=1
gpu=-1
#optim_pose=False
optim_pose=True

exp_name=${exp_name}'-'${voxel_size}

# example:
#./train.sh -d ${scene} -l ${exp_name} --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio}
if [ "$optim_pose" = True ]; then
    ./train_barf.sh -d ${scene} -l ${exp_name} --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --optim_pose ${optim_pose}
else
    ./train_barf.sh -d ${scene} -l ${exp_name} --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio}
fi

