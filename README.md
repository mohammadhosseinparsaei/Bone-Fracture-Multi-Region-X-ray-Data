# Medical Image Classification with Deep Learning
This project aims to detect bone fractures in X-ray images. Deep learning techniques implemented using the **PyTorch** library are utilized for this classification task.
### Introduction
Medical imaging plays a crucial role in diagnosing various conditions, including bone fractures. With the advent of deep learning, automated classification of medical images has become feasible, aiding healthcare professionals in accurate and timely diagnosis.
### Dataset
The dataset used in this project consists of medical images depicting bones, with annotations indicating whether each image contains a fracture or not. The dataset is structured into three main directories: `train`, `val`, and `test`. Each of these directories contains two subdirectories: `fractured` and `not fractured`, corresponding to images with and without fractures, respectively.
### Methodology
Deep learning models are trained on the provided dataset using PyTorch, a popular deep learning framework. Convolutional neural networks (CNNs) are employed to learn discriminative features from the images.
### Training on GPU
Despite the modest hardware specifications (e.g., MX130 GPU), efforts have been made to accelerate model training by utilizing the GPU. By installing PyTorch with GPU support, the training process can leverage the computational power of the GPU, resulting in faster iterations and reduced training times.
### Data Access
The data is accessible and downloadable from [here](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data/data).
**Links to Original Datasets**:
Bone Break Classifier Dataset - [https://www.kaggle.com/datasets/amohankumar/bone-break-classifier-dataset]
bone_fracture Dataset - [https://www.kaggle.com/datasets/abdelazizfaramawy/bone-fracture]
fracture Dataset - [https://kaggle.com/datasets/harshaarya/fracture]

### Tools and Libraries Used
- Python 3.11.7
- numpy 1.24.3
- pillow 10.2.0
- scikit-learn 1.3.0
- pytorch 2.2.2 + cu 118
- matplotlib 3.8.4
### An example of images
![images](https://github.com/mohammadhosseinparsaei/Bone-Fracture-Multi-Region-X-ray-Data/blob/main/sample_images.png)
### Model Performance
#### Accuracy & Loss Plot
![Accuracy & Loss plot](https://github.com/mohammadhosseinparsaei/Bone-Fracture-Multi-Region-X-ray-Data/blob/main/train_val_loss_acc_plot.png)

