import tensorflow.keras as keras
import keras.models
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Dropout, Dense
import numpy as np
class MRNet_VGG(keras.Model):
  def __init__(self):
    super(MRNet_VGG, self).__init__()

    self.inputs = Input(shape=(224,224,3));

    self.conv1_1 = Conv2D(filters=64,kernel_size=(3,3), padding="same", activation="relu")(self.inputs)
    self.conv1_2 = Conv2D(filters=64,kernel_size=(3,3), padding="same", activation="relu")(self.conv1_1)
    self.max_pooling1 = MaxPool2D(pool_size=(2,2),strides=(2,2))(self.conv1_2)


    self.conv2_1 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(self.max_pooling1)
    self.conv2_2 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(self.conv2_1)
    self.max_pooling2 = MaxPool2D(pool_size=(2,2),strides=(2,2))(self.conv2_2)


    self.conv3_1 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(self.max_pooling2)
    self.conv3_2 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(self.conv3_1)
    self.conv3_3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(self.conv3_2)
    self.max_pooling3 = MaxPool2D(pool_size=(2,2),strides=(2,2))(self.conv3_3)


    self.conv4_1 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(self.max_pooling3)
    self.conv4_2 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(self.conv4_1)
    self.conv4_3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(self.conv4_2)
    self.max_pooling4 = MaxPool2D(pool_size=(2,2),strides=(2,2))(self.conv4_3)


    self.conv5_1 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(self.max_pooling4)
    self.conv5_2 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(self.conv5_1)
    self.conv5_3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(self.conv5_2)
    self.max_pooling5 = MaxPool2D(pool_size=(2,2),strides=(2,2))(self.conv5_3)


    self.avg_pooling = AveragePooling2D(pool_size=(7, 7), strides=None, padding="same")
    self.avg_pooling.build((None, 512, 7, 7))

    self.dropout=Dropout(0.5)
    self.dropout.build((512,))
    self.classifier = Dense(1, input_shape=(512,), activation="sigmoid")
    self.classifier.build((None,512))

  def _VGG(self, inputs):
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


  def call(self, inputs):
    outputs = np.empty((0))
    for exam in inputs:
      out = np.empty((0, 512, 7, 7))
      for scan in exam:
        out = np.append(out, np.expand_dims(_VGG(scan), axis=0), axis=0)

      out = np.squeeze(self.avg_pooling(out))
      max_idx = np.argmax(out, axis=0)
      out = out[max_idx, range(max_idx.size)]

      out = self.classifier(self.dropout(out))
      np.append(outputs, out, axis=0)
    return outputs




