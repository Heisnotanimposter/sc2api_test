import keras  # Keras 2.1.2 and TF-GPU 1.9.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import os
import random

def __init__(self, use_model=False):
    self.ITERATIONS_PER_MINUTE = 165
    self.MAX_WORKERS = 50
    self.do_something_after = 0
    self.use_model = use_model

    self.train_data = []
    if self.use_model:
        print("USING MODEL!")
        self.model = keras.models.load_model("BasicCNN-30-epochs-0.0001-LR-4.2")