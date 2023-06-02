# Skinned Motion Retargeting with Residual Perception of Motion Semantics & Geometry

This is the code for the CVPR 2023 paper [Skinned Motion Retargeting with Residual Perception of Motion Semantics & Geometry](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Skinned_Motion_Retargeting_With_Residual_Perception_of_Motion_Semantics__CVPR_2023_paper.html) by Jiaxu Zhang, et al.

R2ET is a neural motion retargeting model that can preserve source motion semantics and avoid interpenetration in target motion.

![](https://github.com/Kebii/R2ET/gifs/demo1.gif)


## Quick Start
### 1. Conda environment
```
conda create python=3.9 --name r2et
conda activate r2et
```

### 2. Install dependencies (Anaconda installation is recommended)
Install the packages in `requirements.txt` and install [PyTorch 1.10.0](https://pytorch.org/)
```
pip install -r requirements.txt
```

Install pytorch
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
```

### 3. Download and Install Blender
**Download and install from:**  
* https://www.blender.org/download/


### 4. Data preparation
**Train data:**  
* Firstly, create an account in the [Mixamo](https://www.mixamo.com) website.
* Next, download the fbx animation files for each character folder in ./datasets/mixamo/train_char/. The animation list can be refered to [NKN](https://github.com/rubenvillegas/cvpr2018nkn). we collect 1952 non-overlapping motion sequences for training.
Once the fbx files have been downloaded, run the following ***blender script*** to convert them into BVH files:
```
blender -b -P ./datasets/fbx2bvh.py
```
* Finally, preprocess the bvh files into npy files by running the following command:
```
python ./datasets/preprocess_q.py
```

* The shape information saved in ./datasets/mixamo/train_shape (already preprocessed) for each character's T-pose is preprocessed by:
```
blender -b -P ./datasets/extract_shape.py
```

### 5. Install CUDA implementation of SDF function:
```
cd ./outside-code/sdf
python setup.py install
```

## Inference
**NKN Autoencoder:**
```
python3 inference_bvh.py --config ./config/inference_bvh_cfg.yaml
```

## Training
**Skeleton-aware Network:**
```
python3 train_skeleton_aware.py --config ./config/train_skeleton_aware.yaml
```

**Shape-aware Network:**
```
python3 train_shape_aware.py --config ./config/train_shape_aware.yaml
```

## visualization
```
cd ./visualization
blender -P visualize.py
```

## Citation                                                                                                                                                  
* If you find this useful, please cite our work as follows:                        
```                                                                              
@inproceedings{zhang2023skinned,
  title     = {Skinned Motion Retargeting with Residual Perception of Motion Semantics & Geometry},
  author    = {Jiaxu Zhang, Junwu Weng, Di Kang, Fang Zhao, Shaoli Huang, Xuefei Zhe, Linchao Bao, Ying Shan, Jue Wang, Zhigang Tu.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023}
  }
```