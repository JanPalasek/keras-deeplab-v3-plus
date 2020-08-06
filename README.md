# Keras implementation of Deeplabv3+  
DeepLab is a state-of-art deep learning model for semantic image segmentation.  

Model is based on the original TF frozen graph. It is possible to load pretrained weights into this model. Weights are directly imported from original TF checkpoint.  

Segmentation results of original TF model. __Output Stride = 8__
<p align="center">
    <img src="imgs/seg_results1.png" width=600></br>
    <img src="imgs/seg_results2.png" width=600></br>
    <img src="imgs/seg_results3.png" width=600></br>
</p>

Segmentation results of this repo model with loaded weights and __OS = 8__  
Results are identical to the TF model  
<p align="center">
    <img src="imgs/my_seg_results1_OS8.png" width=600></br>
    <img src="imgs/my_seg_results2_OS8.png" width=600></br>
    <img src="imgs/my_seg_results3_OS8.png" width=600></br>
</p>

Segmentation results of this repo model with loaded weights and __OS = 16__  
Results are still good
<p align="center">
    <img src="imgs/my_seg_results1_OS16.png" width=600></br>
    <img src="imgs/my_seg_results2_OS16.png" width=600></br>
    <img src="imgs/my_seg_results3_OS16.png" width=600></br>
</p>

### How to get labels
Model will return tensor of shape `(batch_size, height, width, num_classes)`. To obtain labels, you need to apply argmax to logits at exit layer. 

### How to use this model with custom input shape and custom number of classes
```python
from deeplab.model import Deeplabv3
deeplab_model = Deeplabv3(input_shape=(384, 384, 3), classes=4)  
```
Alternatively you can use None as shape
```python
deeplab_model = Deeplabv3(input_shape=(None, None, 3), classes=4)
```
After that you will get a usual Keras model which you can train using `.fit` and `.fit_generator` methods.

### How to train this model

Useful parameters can be found in the [original repository](https://github.com/tensorflow/models/blob/master/research/deeplab/train.py).

Important notes:
1. This model doesn’t provide default weight decay, user needs to add it themselves.
2. Due to huge memory use with `OS=8`, Xception backbone should be trained with `OS=16` and only inferenced with `OS=8`.
3. User can freeze feature extractor for Xception backbone (first 356 layers) and only fine-tune decoder. Right now (March 2019), there is a problem with finetuning Keras models with BN. You can read more about it [here](https://github.com/keras-team/keras/pull/9965).

### Xception vs MobileNetv2
There are 2 available backbones. Xception backbone is more accurate, but has 25 times more parameters than MobileNetv2. 

For MobileNetv2 there are pretrained weights only for `alpha=1`. However, you can initiate model with different values of alpha.