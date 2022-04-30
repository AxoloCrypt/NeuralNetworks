import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

tfds.disable_progress_bar()

# print(tfds.list_builders())

builder = tfds.builder('lambada')

info = builder.info

print(info)

