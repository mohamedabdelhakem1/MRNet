import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Dropout, Flatten, Conv2D, MaxPool2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta
import numpy as np
import os
import pandas as pd

from tensorflow.keras.initializers import RandomNormal as RN
from tensorflow.keras.initializers import GlorotNormal as GN
from tensorflow.keras.initializers import GlorotUniform as GU

MEAN = 0.0
STDDEV = 0.01
SEED = 5



class AlexNet_block(keras.layers.Layer):

  def __init__(self, input_shape=(227, 227, 3)):
    super(AlexNet_block, self).__init__()

    # 1st Convolutional Layer
    self.conv1 = Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding="valid", activation = "relu")
    # Max Pooling
    self.max_pooling1 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid")

    # 2nd Convolutional Layer
    self.conv2 = Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding="same", activation = "relu")
    # Max Pooling
    self.max_pooling2 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid")

    # 3rd Convolutional Layer
    self.conv3 = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same", activation = "relu")

    # 4th Convolutional Layer
    self.conv4 = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same", activation = "relu")

    # 5th Convolutional Layer
    self.conv5 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation = "relu")
    # Max Pooling
    self.max_pooling5 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid")

  # @tf.function
  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.max_pooling1(x)
    x = self.conv2(x)
    x = self.max_pooling2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.max_pooling5(x)
    return x

class AlexNet_layer(keras.layers.Layer):
  def __init__(self, input_shape, batch_size):
    super(AlexNet_layer, self).__init__()
    self.alexnet = AlexNet_block(input_shape=input_shape[2:])

    # Passing it to a Fully Connected layer
    self.flatten = Flatten()
    # 1st Fully Connected Layer 9216
    self.dense1 = Dense(units = 4096, activation = "relu")
    # Add Dropout to prevent overfitting
    self.dropout1 = Dropout(0.4)

    # 2nd Fully Connected Layer
    self.dense2 = Dense(units = 4096, activation = "relu")
    # Add Dropout
    self.dropout2 = Dropout(0.4)

    # 3rd Fully Connected Layer
    self.dense3 = Dense(4096, activation = "relu")
    # Add Dropout
    self.dropout3 = Dropout(0.4)

    # Output Layer
    self.dense4 = Dense(1, activation = "softmax")

    self.avg_pooling = AveragePooling2D(pool_size=(6, 6), padding="same")
    self.fc = Dense(1, activation="sigmoid", input_dim=512, kernel_initializer=GU(SEED))
    self.dropout = Dropout(0.5)
    self.b_size = batch_size

  @tf.function(autograph=True)                
  def call(self, inputs):
    arr = []
    for index in range(self.b_size):
      out = self.alexnet(inputs[index])
      out = self.avg_pooling(out)
      out = tf.squeeze(out, axis=[1, 2])
      out = keras.backend.max(out, axis=0, keepdims=True)
      out = tf.squeeze(out)
      arr.append(out) 
    output = tf.stack(arr, axis=0)
    output = self.fc(self.dropout(output))
    # print("output shape: ", output.shape)
    return output

  def compute_output_shape(self, input_shape):
    return (None, 1)

def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))

def MRNet_AlexNet_model(batch_size, lr, combination=["abnormal", "axial"]):
  METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
  ]

  b_size = batch_size
  model = keras.Sequential()
  model.add(AlexNet_layer((None, None, 227, 227, 3), b_size))
  model(Input(shape=(None, 227, 227, 3)))
  model.compile(
      optimizer=tf.keras.optimizers.Adam(lr=lr),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=METRICS)
  data_path = "/content/gdrive/My Drive/Colab Notebooks/MRNet/"
  checkpoint_dir = data_path + "training_AlexNet/" + combination[0] + "/" + combination[1] + "/"
  # checkpoint_dir = os.path.dirname(checkpoint_path)
  print("checkpoint_dir: "+checkpoint_dir)
  if not os.path.exists(checkpoint_dir):
    print("make dir: "+checkpoint_dir)
    os.makedirs(checkpoint_dir)
  os.chdir(checkpoint_dir)
  cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_dir + "weights.{epoch:02d}.hdf5", save_weights_only=True, verbose=1)
  tcb = TestCallback(model)
  return model, [cp_callback, tcb]

class TestCallback(tf.keras.callbacks.Callback):
  def __init__(self, model):
    super(TestCallback, self).__init__()
    self.model = model

  def on_epoch_end(self, epoch, logs=None):
    if (epoch == 0):
      self.w = self.model.layers[0].get_weights()[0]
      return
    self.w_after = self.model.layers[0].get_weights()[0]
    print('  TestCallback: ', (self.w == self.w_after).all())
    self.w = self.w_after
