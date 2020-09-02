# Semantic-Segmentation


# Introduction

Semantic Segmentation is an image analysis task in which we classify each pixel in the image into a class. This is similar to what humans do whenever we are looking at something, we try to categorize what portion of the image belongs to which class/label. Humans would categorize larger shapes belonging to one class immediately without looking further at the details, but would look at it more closely in case the classification is not immediate. This approach mixes a global appreciation and a finer examination. This is what FCN models can do as well in the architecture described below.

The most common use cases for the Semantic Segmentation are:
- Autonomous Driving : Segmenting out objects like Cars, Pedestrians, Lanes and traffic signs so that the computer driving the car has a good understanding of the road scene in front of it.
- Facial Segmentation: Segmenting each part of the face into semantically similar regions – lips, eyes etc. This can be useful in many real-world applications. One application can be virtual make-over.
- Indoor Object Segmentation: Can be used combined with AR (Augmented Reality) and VR (Virtual Reality) and useful to architecture, interior design, furniture manufacture and retail shops.
- Geo Land Sensing: Geo Land Sensing is a way of categorising each pixel in satellite images into a category such that we can track the land cover of each area. So, it can be useful to prevent fires. Similarly, if ithere is heavy deforestation taking place n some area then appropriate measures can be taken. There can be many more applications using semantic segmentation on satellite images.

# Project description

In this project, we label the pixels of a road in images using a Fully Convolutional Network (FCN). The principle of FCNs is to remove all fully-connected layers to preserve spacial information and obtain a classification of each pixel or the original image as an output.

I use the Semantic Segmentation network built and adapted after the following [paper](https://arxiv.org/abs/1411.4038) 2015, Jonathan Long et al. The FCN introduced by the authors learns to combine coarse, high layer information with fine, low layer information, nobably by using "skip connections". 

The FCN model architecture comprises :
- an encoder, usually a pre-trained proven network whith all dense, fully_connected, classification section removed. As such the encoder acts as a feature extraction with only convolutional layers keeping the spacial information. The obtained feature volume is passed thru a 1x1 final convolutional layer to obtain a grid. I used VGG16.
- a decoder which will upsample the extracted spacial information back to the original image size using transpose convolutional layers also preserving the spacial information. The output is a grid of size identical to the original image and depth equals to the number of classes. Each pixel is therefore associated to a probability vector.

![](asset/fcn_general.jpg)

- skip connections are intermediate outputs from the encoder fed directly into the decoder at various stages. In the paper "Fully Convolutional Networks for Semantic Segmentation", the authors used a pre-trained VGG16 for the encoder and extracted the output of maxpooling layers 3 and 4 (output of layer blocks 3 and 4) to fed into the decoder along the upsampling process. The output of layer 7 (final maxpooling layer) is pushed through a 1x1 convolutional layer and fed as input to the decoder (commonly refered as the third skip connection).

The principles are summarized in the below sketch: The pooling and prediction layers are shown as grid that reveal relative spatial coarseness, while intermediate layers are shown as vertical lines.

![](asset/fcn.jpg)

My model :
- The encoder: VGG16 model pretrained on ImageNet for classification (see VGG16 architecutre below). The fully-connected layers are replaced by 1-by-1 convolution.

![](asset/vgg16.png)

- The decoder: Transposed convolution is used to upsample the input to the original image size. Skip connections from the encoder to the decoder are used in the model.

![]()

Pretrained models expect a 3-channel image (RGB) which is normalized with the Imagenet mean and standard deviation, i.e., mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]. The input dimension is [N x C x H x W] where,

N -> the batch size
C -> the number of channels (which is 3)
H -> the height of the image
W -> the width of the image

The output dimension of the model is [N x C' x H x W] where,

N -> is the batch size
C' -> is the number of classes set for the classification task
H -> the height of the original image 
W -> the width of the original image 


# Dataset

The model is trained in Pytorch using [KITTI Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) which can be downloaded [here](). The road and lane estimation benchmark consists of 289 training and 290 test images. Ground truth has been generated by manual annotation of the images and is available for two different road terrain types: road - the road area, i.e, the composition of all lanes, and lane - the ego-lane, i.e., the lane the vehicle is currently driving on (only available for category "um"). Ground truth is provided for training images only. For out purpose, we distinguish only two classes: road and non road (background).

![]()


## Results

![](asset/folder_structure.PNG)

