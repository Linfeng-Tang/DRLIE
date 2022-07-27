# DRLIE
This is official Pytorch implementation of "[DRLIE: Flexible Low-Light Image Enhancement via Disentangled Representations](https://ieeexplore.ieee.org/document/9833451)"
## Framework
![A typical schematic of our proposed DRLIE.](https://github.com/Linfeng-Tang/DRLIE/blob/main/DLIE/first_show.jpg)
A typical schematic of our proposed DRLIE.

![Schematic of the proposed custom illumination guided low-light image enhancement algorithm based on disentangle representations.](https://github.com/Linfeng-Tang/DRLIE/blob/main/DLIE/Low-image-Enhance.jpg)
Schematic of the proposed custom illumination guided low-light image enhancement algorithm based on disentangle representations.

![Framework of disentanglement model for multiexposure images.](https://github.com/Linfeng-Tang/DRLIE/blob/main/DLIE/training.jpg)
Framework of disentanglement model for multiexposure images.

![Network architecture of the content encoder, attribute encoder, and generator.](https://github.com/Linfeng-Tang/DRLIE/blob/main/DLIE/network-Rebuttal.jpg)
Network architecture of the content encoder, attribute encoder, and generator.

## Coding
### Recommended Environment
 - [ ] tensorflow-gpu 1.14.0 
 - [ ] scipy 1.2.0   
 - [ ] numpy 1.19.2
 - [ ] opencv 3.4.2 

 ### To Train
 Please load the training dataset from [here](ttps://pan.baidu.com/s/1v9OclGBFX-BTXZTSgfcL9Q?pwd=DRLI) and place it in this project.

	
Then, training the disentanglement model by entering the following code

    CUDA_VISIBLE_DEVICES=0,1 python main.py --phase train --dataset over2under

Moreover, you also could place the underexposed images from the [MEF](https://github.com/csjcai/SICE) dataset in **./dataset/over2under/trainA** 
	and put the overexposed images in **./dataset/over2under/trainB**.
	Then, rewrite the dataloader to load your dataset and retrain your model.
	
**Note**:  the training of the disentanglement model is extremely unstable and may not yields an excellent model. 
### To Test

    CUDA_VISIBLE_DEVICES=0,1 python main.py --phase guide --dataset AGLIE --guide_num 129 --batch_size 1 --direction a2b
The test images from the AGLIE datset please put in ./dataset/AGLIE/ and the guided images please put in ./guide/ .  You can modify the relevant configureation according to our producedures.

## Experiment Results
![Visual results of different low-light image enhancement methods on the AGLIE dataset.](https://github.com/Linfeng-Tang/DRLIE/blob/main/DLIE/figure_AGLIE.jpg)
Visual results of different low-light image enhancement methods on the AGLIE dataset.

![Visual results of different low-light image enhancement methods on the MEF dataset.](https://github.com/Linfeng-Tang/DRLIE/blob/main/DLIE/figure_MEF.jpg)
Visual results of different low-light image enhancement methods on the MEF dataset.

![Visual results of different low-light image enhancement methods on the LOL dataset.](https://github.com/Linfeng-Tang/DRLIE/blob/main/DLIE/figure_LOL.jpg)
Visual results of different low-light image enhancement methods on the LOL dataset.

![Visual results of different low-light image enhancement methods on the VV dataset.](https://github.com/Linfeng-Tang/DRLIE/blob/main/DLIE/figure_VV.jpg)
Visual results of different low-light image enhancement methods on the VV dataset.

![Example of custom illumination adjustment with specific exposure levels.](https://github.com/Linfeng-Tang/DRLIE/blob/main/DLIE/gamma_guide.jpg)
Example of custom illumination adjustment with specific exposure levels.

![Visual results of controllable illumination manipulation with multiexposure guide images.](https://github.com/Linfeng-Tang/DRLIE/blob/main/DLIE/Exposure_guide.jpg)
Visual results of controllable illumination manipulation with multiexposure guide images.

![Some typical examples of flexible illumination adjustment, guided by the MIT-Adobe FiveK dataset.](https://github.com/Linfeng-Tang/DRLIE/blob/main/DLIE/Expert_Guide.jpg)
Some typical examples of flexible illumination adjustment, guided by the MIT-Adobe FiveK dataset.

## Citation
```
@article{Tang2022DRLIE,
  author={Tang, Linfeng and Ma, Jiayi and Zhang, Hao and Guo, Xiaojie},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={DRLIE: Flexible Low-Light Image Enhancement via Disentangled Representations}, 
  year={2022},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TNNLS.2022.3190880}}
```

## Acknowledgement
The codes are heavily based on [DRIT-Tensorflow](https://github.com/taki0112/DRIT-Tensorflow). Please also follow their licenses. Thanks for their awesome works.
