# Semantic-Segmentation

Semantic Segmentation is an image analysis task in which we classify each pixel in the image into a class. This is similar to what us humans do all the time by default. Whenever we are looking at something, then we try to “segment” what portion of the image belongs to which class/label/category.

In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

The road and lane estimation benchmark consists of 289 training and 290 test images. Ground truth has been generated by manual annotation of the images and is available for two different road terrain types: road - the road area, i.e, the composition of all lanes, and lane - the ego-lane, i.e., the lane the vehicle is currently driving on (only available for category "um"). Ground truth is provided for training images only.

you'll notice that layers 3, 4 and 7 of VGG16 are utilized in creating skip layers for a fully convolutional network. The reasons for this are contained in the paper Fully Convolutional Networks for Semantic Segmentation.

The most common use cases for the Semantic Segmentation are:
- Autonomous Driving : Segmenting out objects like Cars, Pedestrians, Lanes and traffic signs so that the computer driving the car has a good understanding of the road scene in front of it.
- Facial Segmentation: Segmenting each part of the face into semantically similar regions – lips, eyes etc. This can be useful in many real-world applications. One application can be virtual make-over.
- Indoor Object Segmentation: Can be used combined with AR (Augmented Reality) and VR (Virtual Reality) and useful to architecture, interior design, furniture manufacture and retail shops.
- Geo Land Sensing: Geo Land Sensing is a way of categorising each pixel in satellite images into a category such that we can track the land cover of each area. So, it can be useful to prevent fires. Similarly, if ithere is heavy deforestation taking place n some area then appropriate measures can be taken. There can be many more applications using semantic segmentation on satellite images.

 We will use the COCO (Common Objects in Context) image dataset for Semantic Image Segmentation in Python with libraries including PyCoco, and Tensorflow Keras.
“COCO is a large-scale object detection, segmentation, and captioning dataset.”
COCO provides multi-object labeling, segmentation mask annotations, image captioning, key-point detection and panoptic segmentation annotations with a total of 81 categories, making it a very versatile and multi-purpose dataset.


These models expect a 3-channel image (RGB) which is normalized with the Imagenet mean and standard deviation, i.e.,
mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]

the input dimension is [Ni x Ci x Hi x Wi]
where,

Ni -> the batch size
Ci -> the number of channels (which is 3)
Hi -> the height of the image
Wi -> the width of the image
And the output dimension of the model is [No x Co x Ho x Wo]
where,

No -> is the batch size (same as Ni)
Co -> is the number of classes that the dataset have!
Ho -> the height of the image (which is the same as Hi in almost all cases)
Wo -> the width of the image (which is the same as Wi in almost all cases)



NOTE: The output of torchvision models is an OrderedDict and not a torch.Tensor
And in during inference (.eval() mode ) the output, which is an OrderedDict just has one key – out. This out key holds the output and it’s corresponding value has the shape of [No x Co x Ho x Wo].



The Semantic Segmentation network provided by this paper learns to combine coarse, high layer informaiton with fine, low layer information. The pooling and prediction layers are shown as grid that reveal relative spatial coarseness, while intermediate layers are shown as vertical lines

The encoder
VGG16 model pretrained on ImageNet for classification (see VGG16 architecutre below) is used in encoder.
And the fully-connected layers are replaced by 1-by-1 convolutions.
The decoder
Transposed convolution is used to upsample the input to the original image size.
Two skip connections are used in the model.

![](asset/fcn_general.jpg)

![](asset/fcn.jpg)

![](asset/vgg16.png)


# Manipulate-COCO-dataset
Explore COCO dataset and manipulate elements in the context of semantic segmentation

This notebook explores the COCO (Common Objects in Context) image dataset and can provide helpers functions for Semantic Image Segmentation in Python.

“COCO is a large-scale object detection, segmentation, and captioning dataset.”

COCO provides multi-object labeling, segmentation mask annotations, image captioning, key-point detection and panoptic segmentation annotations with a total of 81 categories, making it a very versatile and multi-purpose dataset.

COCO 2017 dataset comes with nearly 120.000 training images, each with at least 5 captions, pixelwise semantic segmentation, keypoints... as well as 40.670 test images and 5000 validation images.

Visit [](http://cocodataset.org/) for more information on COCO, including for the data, paper, and tutorials. I recommend checking for the exact format of the annotations described on the COCO website [here](https://cocodataset.org/#format-data).

## Requirements
- download COCO API from [here](https://github.com/cocodataset/cocoapi). This package provides Matlab, Python, and Lua APIs that assists in loading, parsing, and visualizing the annotations in COCO. Select yours and replicate the speficic API folder into your working folder.
- download COCO dataset from [here](https://cocodataset.org/#download). I use the 2017 train/Val/test image sets as well as 2017 Trai,/Val annotations in this notebook. A lighter alternative exists with the 2014. Beware that the dataset is extremely large (over 24 Gb of data).
- create a images folder within your working folder to store your own set of images

## Folder structure
I used this folder structure which you can eventually replicate. My working drive where to put the notebook is the top folder.

![](asset/folder_structure.PNG)

