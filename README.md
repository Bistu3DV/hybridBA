# Hybrid bundle-adjusting 3D Gaussians for view consistent rendering with pose optimization


## Installation

We tested on a server configured with Ubuntu 18.04, cuda 11.6 and gcc 9.4.0. Other similar configurations should also work, but we have not verified each one individually.

1. Clone this repo:

```
git clone https://github.com/Bistu3DV/hybridBA.git
cd hybridBA
```

2. Install dependencies

```
SET DISTUTILS_USE_SDK=1 # Windows only, we don't verified on windons
conda env create --file environment.yml
conda activate hybridBA
```

## Data

First, create a ```data/``` folder inside the project path by 

```
mkdir data
```

The data structure will be organised as follows:

```
data/
├── dataset_name
│   ├── scene1/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
│   ├── scene2/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
...
```


### Public Data

The BungeeNeRF dataset is available in [Google Drive](https://drive.google.com/file/d/1nBLcf9Jrr6sdxKa1Hbd47IArQQ_X8lww/view?usp=sharing)/[百度网盘[提取码:4whv]](https://pan.baidu.com/s/1AUYUJojhhICSKO2JrmOnCA). The MipNeRF360 scenes are provided by the paper author [here](https://jonbarron.info/mipnerf360/). And we test on scenes ```bicycle, bonsai, counter, garden, kitchen, room, stump```. The SfM data sets for Tanks&Temples and Deep Blending are hosted by 3D-Gaussian-Splatting [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip). Download and uncompress them into the ```data/``` folder.

### Custom Data

For custom data, you should process the image sequences with [Colmap](https://colmap.github.io/) to obtain the SfM points and camera poses. Then, place the results into ```data/``` folder.


### Training a single scene

For training a single scene, modify the path and configurations in ```single_train.sh``` accordingly and run it, use barf with optim_pose = True:

```
bash ./single_train.sh
```

- scene: scene name with a format of ```dataset_name/scene_name/``` or ```scene_name/```;
- exp_name: user-defined experiment name;
- gpu: specify the GPU id to run the code. '-1' denotes using the most idle GPU. 
- voxel_size: size for voxelizing the SfM points, smaller value denotes finer structure and higher overhead, '0' means using the median of each point's 1-NN distance as the voxel size.
- update_init_factor: initial resolution for growing new anchors. A larger one will start placing new anchor in a coarser resolution.

> For these public datasets, the configurations of 'voxel_size' and 'update_init_factor' can refer to the above batch training script. 


This script will store the log (with running-time code) into ```outputs/dataset_name/scene_name/exp_name/cur_time``` automatically.

For test set pose optimizaiton and Rendering, use ```single_test_opti.sh``` accordingly and run it, use barf with optim_pose = True:

```
bash single_test_opti.sh
```


## Evaluation

We've integrated the rendering and metrics calculation process into the training code. So, when completing training, the ```rendering results```, ```fps``` and ```quality metrics``` will be printed automatically. And the rendering results will be save in the log dir. Mind that the ```fps``` is roughly estimated by 

```
torch.cuda.synchronize();t_start=time.time()
rendering...
torch.cuda.synchronize();t_end=time.time()
```

which may differ somewhat from the original 3D-GS, but it does not affect the analysis.

Meanwhile, we keep the manual rendering function with a similar usage of the counterpart in [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) and [Scaffold-GS](https://github.com/city-super/Scaffold-GS), one can run it by 

```
bash single_metrics.sh # Compute error metrics on renderings
```

## Results of training three datasets
|Datasets       | PSNR | SSIM | LPIPS |  
|--------------|------|------|-------|  
| Tanks&Temples| 25.17| 0.855| 0.164 |  
| CO3D         | 26.43| 0.856| 0.204 |  
| Glasses case | 31.90| 0.934| 0.186 |

## Run Script
Train scene from the Tanks&Temples dataset were used as an example.

1.Change the path to the scene in single_train.sh to the path where the data is stored:
```
scene='tandt/train'
```

2.train
```
bash single_train.sh
```

3.Change the path to the scene in single_test_opti.sh to the path where the data is stored:
```
scene='tandt/train'
```

Replace the model_path of single_test_opti.sh with the path where the output model is stored, which is in the outputs folder.

4.test
```
bash single_test_opti.sh
```

5.Replace the --model_path of single_metrics.sh with the path where the output model is stored, which is in the outputs folder.

6.metrics
```
bash single_metrics.sh
```


## Contact
zhangbk0566@126.com

## Citation
```
@article{guo2024hybrid,
  title={Hybrid bundle-adjusting 3D Gaussians for view consistent rendering with pose optimization},
  author={Guo, Yanan and Xie, Ying and Chang, Ying and Zhang, Benkui and Jia, Bo and Cao, Lin},
  journal={arXiv preprint arXiv:2410.13280},
  year={2024}
}
```

## LICENSE

Please follow the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).

## Acknowledgement

We thank all authors from [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting), [Scaffold-GS](https://github.com/city-super/Scaffold-GS) and [instantSplat](https://github.com/NVlabs/InstantSplat) for presenting such an excellent work.
