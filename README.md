# CSE802 Face Augmentation and Recognition Project

# Dataset
We take CASIA-Webface[1] dataset as our training data for Face recognition model.
The facial landmark aligned dataset is available from [https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view?usp=sharing](insightface)

## Preprocessing

### Training data
We download the dataset from the above link [https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view?usp=sharing](insightface), and extract face aligned rgb images. 

```
unzip faces_webface_112x112.zip
python prepare_training_data.py
```




[1] Dong Yi, Zhen Lei, Shengcai Liao, Stan Z. Li. Learning Face Representation from Scratch. arXiv:1411.7923, 2014.

