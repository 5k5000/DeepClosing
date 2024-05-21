# Deep Closing: Enhancing Topological Connectivity in Medical Tubular Segmentation
The repo of our paper 'Deep Closing: Enhancing Topological Connectivity in Medical Tubular Segmentation'


## Environment
```
conda create -n DeepClosing python=3.10
conda activate DeepClosing
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pytorch-lightning==1.5.0
pip install monai==0.9.0
pip install scikit-image
pip install wandb
pip install nibabel
```

## Quick Reference:
To begin with, the proposed framework,Deep Closing, consists of two operation:
- (1) Deep Dilation: https://github.com/5k5000/DeepClosing/blob/c7c5ce9d3173ae1449f739da1dfc38aac61c4074/DeepClosing.py#L289
- (2) Simple Component Erosion:
https://github.com/5k5000/DeepClosing/blob/2e7cb06a8948dea991485ff989017f9bb5b03977/DeepClosing.py#L259
- (*) DeepClosing = DeepDilation + Simple Component Erosion (Inference):
https://github.com/5k5000/DeepClosing/blob/2e7cb06a8948dea991485ff989017f9bb5b03977/DeepClosing.py#L313


The implementation of the proposed Simple Point Erosion Module is presented in the position below:
https://github.com/5k5000/DeepClosing/blob/2e7cb06a8948dea991485ff989017f9bb5b03977/DeepClosing.py#L313

Besides, the Masked Shape Reconstruction (Training Stage) is presented in the position below:
https://github.com/5k5000/DeepClosing/blob/2e7cb06a8948dea991485ff989017f9bb5b03977/DeepClosing.py#L30


## todo
We plan to provide more detailed information after the acceptance of our paper. Thanks for your constructive comments to help us improve our paper.



