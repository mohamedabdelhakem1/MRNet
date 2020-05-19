import keras
import numpy as np
from keras.layers import Conv2D,Input, MaxPool2D, AveragePooling2D, Dropout, Dense, Flatten
import tensorflow as tf
import os
class VGG_block(keras.layers.Layer):
  def __init__(self, input_shape=(224,224,3)):
    super(VGG_block, self).__init__()
    self.conv1_1 = Conv2D(input_shape=input_shape,filters=64,kernel_size=(3,3), padding="same", activation="relu")
    self.conv1_2 = Conv2D(filters=64,kernel_size=(3,3), padding="same", activation="relu")
    self.max_pooling1 = MaxPool2D(pool_size=(2,2),strides=(2,2))


    self.conv2_1 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")
    self.conv2_2 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")
    self.max_pooling2 = MaxPool2D(pool_size=(2,2),strides=(2,2))


    self.conv3_1 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")
    self.conv3_2 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")
    self.conv3_3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")
    self.max_pooling3 = MaxPool2D(pool_size=(2,2),strides=(2,2))


    self.conv4_1 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")
    self.conv4_2 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")
    self.conv4_3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")
    self.max_pooling4 = MaxPool2D(pool_size=(2,2),strides=(2,2))


    self.conv5_1 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")
    self.conv5_2 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")
    self.conv5_3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")
    self.max_pooling5 = MaxPool2D(pool_size=(2,2),strides=(2,2))
    # self.avg_pooling = AveragePooling2D(pool_size=(7, 7), padding="same")
    # self.model = keras.Model(inputs= self.conv1_1, outputs=self.max_pooling5)
  
  # @tf.function
  def call(self, inputs):
    x = self.conv1_1(inputs)
    x = self.conv1_2(x)
    x = self.max_pooling1(x)
    x = self.conv2_1(x)
    x = self.conv2_2(x)
    x = self.max_pooling2(x)
    x = self.conv3_1(x)
    x = self.conv3_2(x)
    x = self.conv3_3(x)
    x = self.max_pooling3(x)
    x = self.conv4_1(x)
    x = self.conv4_2(x)
    x = self.conv4_3(x)
    x = self.max_pooling4(x)
    x = self.conv5_1(x)
    x = self.conv5_2(x)
    x = self.conv5_3(x)
    x = self.max_pooling5(x)
    return x





class MRNet_vgg_layer(keras.layers.Layer):
  def __init__(self, input_shape, batch_size):
    super(MRNet_vgg_layer, self).__init__()
    self.vgg = VGG_block(input_shape=input_shape[2:])
    self.avg_pooling = AveragePooling2D(pool_size=(7, 7), padding="same")
    self.dropout = Dropout(0.5)
    self.fc = Dense(1, activation="sigmoid", input_dim=512)
    self.b_size = batch_size
    

  @tf.function
  def call(self, inputs):
    arr = []
    for index in range(self.b_size):
      out = self.vgg(inputs[index])
      out = tf.squeeze(self.avg_pooling(out), axis=[1, 2])
      out = keras.backend.max(out, axis=0, keepdims=True)
      out = tf.squeeze(out)
      arr.append(out)
    output = tf.stack(arr, axis=0)
    output = self.fc(self.dropout(output))
    return output

  def compute_output_shape(self, input_shape):
    return (None, 1)


def MRNet_vgg_model(combination = ["abnormal", "axial"]):
  b_size = 1
  model = keras.Sequential()
  model.add(MRNet_vgg_layer((None, None, 224, 224, 3), b_size))
  model(Input(shape=(None, 224, 224, 3)))
  model.compile(optimizer=tf.keras.optimizers.Adam(), loss="binary_crossentropy", metrics=['accuracy'])

  checkpoint_path = "training_vgg/" + combination[0] + "/" + combination[1]+"/cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
  return model, cp_callback











    