# CSE802 Face Augmentation and Recognition Project

# Dataset
We take CASIA-Webface[1] dataset as our training data for Face recognition model.
The facial landmark aligned dataset is available from [insightface](https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view?usp=sharing)

## Preprocessing

### Training data
We download the dataset from the above link [insightface](https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view?usp=sharing)
you might need some dependencies which can be installed with `conda install --file requirements.txt`

You can either prepare the training data yourself, or download it from our shared drive.

1. Prepare the data yourself.
```
cd preprocessing/training_data
unzip faces_webface_112x112.zip
python prepare_training_data.py
```

2. Or download from the link [our shared drive](https://drive.google.com/file/d/1wcpJUrSTmZ-LbqKZi3a0rLTfsg8a6dHk/view?usp=sharing)

## GAN Generate Images
We use the code provided at [DiscoFaceGAN](https://github.com/microsoft/DiscoFaceGAN) [2] to generate the augmented GAN images. The dependencies required to run the code are listed on their Github link provided under "Testing Requirements". The released paper for this work is also available at this [link](https://arxiv.org/abs/2004.11660v2).

# Reference
[1] Dong Yi, Zhen Lei, Shengcai Liao, Stan Z. Li. Learning Face Representation from Scratch. arXiv:1411.7923, 2014.
[2] Yu Deng, Jiaolong Yang, Dong Chen, Fang Wen, and Xin Tong. Disentangled and controllable face image generation via 3d imitative-contrastive learning. CoRR, abs/2004.11660, 2020.
