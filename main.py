# main file to implement the CNN model on the existing dataset
# this is anurag chaudhary
# This is Aditya Tomar
# This is Aditya Panwar

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')


# encoder
encoder = OneHotEncoder()
encoder.fit([[0], [1]])  # 0- tumor 1-Normal

# importing the images data
# This cell updates result list for images with tumor
data = []
paths = []
result = []

# content from google drive aaaar.......@gmail.com
for r, d, f in os.walk(r'/content/drive/MyDrive/ARK/2'):
    for file in f:
        if '.bmp' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128, 128))
    img = np.array(img)
    if(img.shape == (128, 128, 3)):
        data.append(np.array(img))
        result.append(encoder.transform([[0]]).toarray())
