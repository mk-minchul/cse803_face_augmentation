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


# Reference
[1] Dong Yi, Zhen Lei, Shengcai Liao, Stan Z. Li. Learning Face Representation from Scratch. arXiv:1411.7923, 2014.

