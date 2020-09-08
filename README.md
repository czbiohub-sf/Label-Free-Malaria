# Label-Free-Malaria
Image processing software for the Label-free malaria imaging project.

## Introduction
This repo contains Matlab code for the label-free malaria imaging project. In particular, an image analysis pipeline is implemented in order to segment and classify red blood cell images into four categories: [healthy, ring, trophozoite, and schizont]. The last three categories are three blood stages of the parasite we trained our networks to recognize. The pipeline was created in order to train and validate image classifiers at multiple wavelengths and focal slices in order to evaluate performance as a function of these variables. The pipeline allows for direct comparison, ensuring that the same physical RBCs are aligned and labeled identically for all conditions prior to training and validation. 

## How it works
![MultiwavelengthSchema](https://github.com/czbiohub/Label-Free-Malaria/blob/master/Images/Multi-wavelength%20detection%20schemes.png)

## Requirements
- Matlab 2019b+
- Computer Vision Toolbox
- Deep Learning Toolbox
- Deep Learning Toolbox Model for selected network (Only GoogLeNet has been implemented)
