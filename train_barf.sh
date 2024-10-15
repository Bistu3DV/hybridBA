function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

port=$(rand 10000 30000)

lod=0
#lod=8
iterations=30_000
warmup="False"
optim_pose="False"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--logdir) logdir="$2"; shift ;;
        -d|--data) data="$2"; shift ;;
        --lod) lod="$2"; shift ;;
        --gpu) gpu="$2"; shift ;;
        --warmup) warmup="$2"; shift ;;
        --voxel_size) vsize="$2"; shift ;;
        --update_init_factor) update_init_factor="$2"; shift ;;
        --appearance_dim) appearance_dim="$2"; shift ;;
        --ratio) ratio="$2"; shift ;;
        --optim_pose) optim_pose="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

time=$(date "+%Y-%m-%d_%H:%M:%S")

echo ${optim_pose}
if [ "$warmup" = "True" ]; then
    if [ "$optim_pose" = "True" ]; then
        python train_joint.py --eval -s data/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time --optim_pose
    else
        python train_joint.py --eval -s data/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
    fi
else
    if [ "$optim_pose" = "True" ]; then
        python train_joint.py --eval -s data/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time --optim_pose
    else
        python train_joint.py --eval -s data/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
    fi
fi
