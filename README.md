# Skinned Motion Retargeting with Residual Perception of Motion Semantics & Geometry

This is the code for the CVPR 2023 paper [Skinned Motion Retargeting with Residual Perception of Motion Semantics & Geometry](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Skinned_Motion_Retargeting_With_Residual_Perception_of_Motion_Semantics__CVPR_2023_paper.html) by Jiaxu Zhang, et al.

R2ET is a neural **motion retargeting** model that can preserve the source motion semantics and avoid interpenetration in the target motion.

![](https://github.com/Kebii/R2ET/blob/master/gifs/demo1.gif)


## Quick Start
### 1. Conda environment
```
conda create python=3.9 --name r2et
conda activate r2et
```

### 2. Install dependencies (Anaconda installation is recommended)
* Install the packages in `requirements.txt` and install [PyTorch 1.10.0](https://pytorch.org/)
```
pip install -r requirements.txt
```

* Install pytorch
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
```

### 3. Download and Install Blender
* Download and install from: https://www.blender.org/download/


### 4. Data preparation
**Training data:**  
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

### 5. Install the CUDA implementation of SDF function:
```
cd ./outside-code/sdf
python setup.py install
```

## Inference
**Performing inference using bvh files:**
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
* If you find this work helpful, please consider citing it as follows:                    
```                                                                              
@inproceedings{zhang2023skinned,
  title={Skinned Motion Retargeting with Residual Perception of Motion Semantics \& Geometry},
  author={Zhang, Jiaxu and Weng, Junwu and Kang, Di and Zhao, Fang and Huang, Shaoli and Zhe, Xuefei and Bao, Linchao and Shan, Ying and Wang, Jue and Tu, Zhigang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13864--13872},
  year={2023}
}
```

## Acknowledgments
Thanks to [PMnet](https://github.com/ljin0429/bmvc19_pmnet), [SAN](https://github.com/DeepMotionEditing/deep-motion-editing) and [NKN](https://github.com/rubenvillegas/cvpr2018nkn), our code is partially borrowing from them.