import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import imagepreprocess
from model_fitting import modelFitting

train = "Face-Mask-Detection/train"
test = "Face-Mask-Detection/test"
val = "Face-Mask-Detection/val"

# Loading the images and processing them
train_generator, validation_generator = imagepreprocess(train, val)

# training the model
try:
    history = modelFitting(train_generator, validation_generator)
except Exception as e:
    print(e)