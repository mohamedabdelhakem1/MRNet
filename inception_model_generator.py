import keras
import tensorflow as tf
import numpy as np 
from keras.models import Model
from keras.layers.core import Layer
from keras.layers.core import Layer
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input, concatenate, GlobalAveragePooling2D, AveragePooling2D, Flatten, BatchNormalization
from keras.utils import np_utils
import os
from keras.metrics import categorical_accuracy,binary_accuracy
from keras.utils.vis_utils import plot_model




class inceptionV3():
  def __init__(self,shape ,weightsPath = None):
    self.shape =shape;
    self.weightsPath = weightsPath;

  def ConvBatchNorm(self,filters,kernel_size,padding,activation,bias_initializer="zeros",kernel_initializer="glorot_uniform",strides=(1,1)):
    def inp(input):
      conv = Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,activation=activation,bias_initializer= bias_initializer,kernel_initializer=kernel_initializer)(input);
      return BatchNormalization()(conv)
    return inp;

  def inceptionModlueA(self ,inp,filter1_1x1,filter2_pool,filter3_1x1,filter3_3x3,filter4_1x1,filter4_3x3,filter4_3x3_2,kernel_init="glorot_uniform",
    bias_init="zeros",name=None):
    conv1  = self.ConvBatchNorm(filters=filter1_1x1,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(inp)
  
    conv2_1  = AveragePooling2D((3,3),strides=(1,1),padding='same')(inp)
    conv2_2 =  self.ConvBatchNorm(filters=filter2_pool,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(conv2_1)
    
    conv3_1 = self.ConvBatchNorm(filters=filter3_1x1,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(inp)
    conv3_2 = self.ConvBatchNorm(filters=filter3_3x3,kernel_size=(3,3),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(conv3_1)
    
    conv4_1 = self.ConvBatchNorm(filters=filter4_1x1 , kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(inp)
    conv4_2 = self.ConvBatchNorm(filters= filter4_3x3 ,kernel_size=(3,3),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(conv4_1)
    conv4_3 = self.ConvBatchNorm(filters= filter4_3x3_2 ,kernel_size=(3,3),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(conv4_2)
    
    output = concatenate([conv1, conv2_2, conv3_2, conv4_3], axis=3, name=name)
    return output

  def inceptionModuleB(self,inp,filter1_1x1,filter2_pool,filter3_1x1,filter3_1xn,filter3_nx1,filter4_1x1,filter4_1xn,filter4_nx1,filter4_1xn_1,filter4_nx1_2,kernel_init="glorot_uniform",
    bias_init="zeros",name=None):
    conv1  = self.ConvBatchNorm(filters=filter1_1x1,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(inp)

    conv2_1  = AveragePooling2D((3,3),strides=(1,1),padding='same')(inp)
    conv2_2 =  self.ConvBatchNorm(filters=filter2_pool,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(conv2_1)
    
    conv3_1 = self.ConvBatchNorm(filters=filter3_1x1,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(inp)
    conv3_2 = self.ConvBatchNorm(filters=filter3_1xn,kernel_size=(1,7),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(conv3_1)
    conv3_3 = self.ConvBatchNorm(filters=filter3_nx1,kernel_size=(7,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(conv3_2)
    
    conv4_1 = self.ConvBatchNorm(filters=filter4_1x1,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(inp)
    conv4_2 = self.ConvBatchNorm(filters=filter4_1xn,kernel_size=(1,7),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(conv4_1)
    conv4_3 = self.ConvBatchNorm(filters=filter4_nx1,kernel_size=(7,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(conv4_2)
    conv4_4 = self.ConvBatchNorm(filters=filter4_1xn_1,kernel_size=(1,7),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(conv4_3)
    conv4_5 = self.ConvBatchNorm(filters=filter4_nx1_2,kernel_size=(7,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(conv4_4)
    
    output = concatenate([conv1, conv2_2, conv3_3, conv4_5], axis=3, name=name)
    return output
    
  def inceptionModuleC(self,inp,filter1_1x1,filter2_1x1,filter3_1x1,filter3_1x3,filter3_3x1,filter4_1x1,filter4_3x3,filter4_1x3,filter4_3x1,kernel_init="glorot_uniform",
    bias_init="zeros",name=None):
    conv1  = self.ConvBatchNorm(filters=filter1_1x1,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(inp)

    conv2_1  = AveragePooling2D((3,3),strides=(1,1),padding='same')(inp)
    conv2_2 =  self.ConvBatchNorm(filters=filter2_1x1,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(conv2_1)
    
    conv3_1 = self.ConvBatchNorm(filters=filter3_1x1,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(inp)
    conv3_2 = self.ConvBatchNorm(filters=filter3_1x3,kernel_size=(1,3),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(conv3_1)
    conv3_3 = self.ConvBatchNorm(filters=filter3_3x1,kernel_size=(3,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(conv3_1)
    conv_3 = concatenate([conv3_2,conv3_3],axis=3);
    
    conv4_1 = self.ConvBatchNorm(filters=filter4_1x1 , kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(inp)
    conv4_2 = self.ConvBatchNorm(filters= filter4_3x3 ,kernel_size=(3,3),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(conv4_1)
    conv4_3 = self.ConvBatchNorm(filters= filter4_1x3 ,kernel_size=(1,3),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(conv4_2)
    conv4_4 = self.ConvBatchNorm(filters= filter4_3x1 ,kernel_size=(3,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(conv4_2)
    conv_4 = concatenate([conv4_3,conv4_4],axis=3);

    output = concatenate([conv1, conv2_2,conv_3,conv_4], axis=3, name=name)
    return output

  def inceptionModlueD(self,x,filter1_3x3 , filter2_1x1, filter2_3x3 ,  filter2_3x3_2, name=None):
    conv1 = self.ConvBatchNorm(filters=filter1_3x3, kernel_size=(3,3),strides=(2,2),padding='valid',activation='relu')(x);

    conv2_1 = self.ConvBatchNorm(filters=filter2_1x1, kernel_size=(1,1),padding='same',activation='relu')(x);
    conv2_2 = self.ConvBatchNorm(filters=filter2_3x3, kernel_size=(3,3),padding='same',activation='relu')(conv2_1);
    conv2_3 = self.ConvBatchNorm(filters=filter2_3x3_2, kernel_size=(3,3),strides=(2,2),padding='valid',activation='relu')(conv2_2);

    conv3 = MaxPool2D(pool_size=(3,3) , strides=(2,2),padding='valid')(x);

    output = concatenate([conv1, conv2_3,conv3], axis=3, name=name)
    return output
  def inceptionModlueE(self,x,filter1_1x1,filter1_3x3 , filter2_1x1, filter2_1x7 ,filter2_7x1,filter2_3x3,name=None):
    conv1_1 = self.ConvBatchNorm(filters=filter1_1x1,kernel_size=(1,1),padding='same',activation='relu')(x);
    conv1_2 = self.ConvBatchNorm(filters=filter1_3x3,kernel_size=(3,3),strides=(2,2),padding='valid',activation='relu')(conv1_1);
    
    conv2_1 =  self.ConvBatchNorm(filters=filter2_1x1,kernel_size=(1,1),padding='same',activation='relu')(x);
    conv2_2 =  self.ConvBatchNorm(filters=filter2_1x7,kernel_size=(1,7),padding='same',activation='relu')(conv2_1);
    conv2_3 =  self.ConvBatchNorm(filters=filter2_7x1,kernel_size=(7,1),padding='same',activation='relu')(conv2_2);
    conv2_4 =  self.ConvBatchNorm(filters=filter2_3x3,kernel_size=(3,3),strides=(2,2),padding='valid',activation='relu')(conv2_3);
    
    conv3 = MaxPool2D(pool_size=(3,3),strides=2,padding='valid')(x);
    output = concatenate([conv1_2, conv2_4,conv3], axis=3, name=name)
    return output

  def getModel(self):
    input = Input(shape=self.shape);
    # 299 x 299 x 3
    x = self.ConvBatchNorm(filters=32,kernel_size=(3,3),strides=(2,2),padding='valid',activation='relu')(input)
    # 149 x 149 x 32
    x = self.ConvBatchNorm(filters=32,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu')(x)
    # 147 x 147 x 32
    x = self.ConvBatchNorm(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(x)  
    # 147 x 147 x64
    x = MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
    # 73 x 73 x 64
    x = self.ConvBatchNorm(filters=80,kernel_size=(1,1),strides=(1,1),padding='valid',activation='relu')(x)
    # 73 x 73 x 80
    x = self.ConvBatchNorm(filters=192,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu')(x)
    # 71 x 71 x 192    
    x = MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
    # 35 x 35 x192
    x = self.inceptionModlueA(x,filter1_1x1=64,filter2_pool=32,filter3_1x1=48,filter3_3x3=64,filter4_1x1=64,filter4_3x3=96,filter4_3x3_2=96);
    # 35 x 35 x 256
    x = self.inceptionModlueA(x,filter1_1x1=64,filter2_pool=64,filter3_1x1=48,filter3_3x3=64,filter4_1x1=64,filter4_3x3=96,filter4_3x3_2=96);
    # 35 x 35 x288
    x = self.inceptionModlueA(x,filter1_1x1=64,filter2_pool=64,filter3_1x1=48,filter3_3x3=64,filter4_1x1=64,filter4_3x3=96,filter4_3x3_2=96);
    # 35 x 35 x 288
    x = self.inceptionModlueD(x,filter1_3x3=384 , filter2_1x1=64, filter2_3x3=96 ,  filter2_3x3_2=96);
    # 17 x 17 x 768
    x =  self.inceptionModuleB(x,filter1_1x1=192,filter2_pool=192,filter3_1x1=128,filter3_1xn=128,filter3_nx1=192,filter4_1x1=128,filter4_1xn=128,filter4_nx1=128
                          ,filter4_1xn_1=128,filter4_nx1_2=192);
    # 17 x 17 x 768
    x =  self.inceptionModuleB(x,filter1_1x1=192,filter2_pool=192,filter3_1x1=128,filter3_1xn=128,filter3_nx1=192,filter4_1x1=128,filter4_1xn=128,filter4_nx1=128
                          ,filter4_1xn_1=128,filter4_nx1_2=192);
     # 17 x 17 x 768
    x =  self.inceptionModuleB(x,filter1_1x1=192,filter2_pool=192,filter3_1x1=160,filter3_1xn=160,filter3_nx1=192,filter4_1x1=160,filter4_1xn=160,filter4_nx1=160
                          ,filter4_1xn_1=160,filter4_nx1_2=192);
     # 17 x 17 x 768      
    x =  self.inceptionModuleB(x,filter1_1x1=192,filter2_pool=192,filter3_1x1=160,filter3_1xn=160,filter3_nx1=192,filter4_1x1=160,filter4_1xn=160,filter4_nx1=160
                          ,filter4_1xn_1=160,filter4_nx1_2=192);
     # 17 x 17 x 768
    x =  self.inceptionModuleB(x,filter1_1x1=192,filter2_pool=192,filter3_1x1=192,filter3_1xn=192,filter3_nx1=192,filter4_1x1=192,filter4_1xn=192,filter4_nx1=192
                          ,filter4_1xn_1=192,filter4_nx1_2=192);
     # 17 x 17 x 768
    x1 = x;

    x1 = AveragePooling2D((5, 5), strides=3,padding='valid')(x1)
    # 5 x 5 x 768
    x1 = self.ConvBatchNorm(128, (1, 1), padding='same', activation='relu')(x1)
   

    x = self.inceptionModlueE(x,filter1_1x1=192,filter1_3x3=320 , filter2_1x1=192, filter2_1x7=192 ,filter2_7x1 =192 ,filter2_3x3=192);

    x = self.inceptionModuleC(x,filter1_1x1=320,filter2_1x1=192,filter3_1x1=384,filter3_1x3=384,filter3_3x1=384,filter4_1x1=448,filter4_3x3=384,filter4_1x3=384,filter4_3x1=384);

    x = self.inceptionModuleC(x,filter1_1x1=320,filter2_1x1=192,filter3_1x1=384,filter3_1x3=384,filter3_3x1=384,filter4_1x1=448,filter4_3x3=384,filter4_1x3=384,filter4_3x1=384);
  
  
    model = Model(inputs=input, outputs =[x, x1], name='inception_v3')
    # model.summary()
    
    if self.weightsPath:
        model.load_weights(self.weightsPath)
    return model


class MRNet_inception_layer(keras.layers.Layer):
  def __init__(self, batch_size):
    super(MRNet_inception_layer, self).__init__()
    self.inception = inceptionV3((299, 299, 3)).getModel()
    self.avg_pooling1 = AveragePooling2D(pool_size=(8, 8), padding="same")
    self.avg_pooling2 = AveragePooling2D(pool_size=(5, 5), padding="same")
    self.d1 = Dropout(0.5)
    self.d2 = Dropout(0.5)

    self.fc1 = Dense(1, activation="sigmoid", input_dim=2048)
    self.fc2 = Dense(1, activation="sigmoid", input_dim=128)
    self.b_size = batch_size

  def compute_output_shape(self, input_shape):
    return (None, 2)

  def call(self, inputs):
    arr1 = []
    arr2 = []
    for index in range(self.b_size):
      f_list = self.inception(inputs[index])
      out1 = tf.squeeze(self.avg_pooling1(f_list[0]), axis=[1, 2])
      out1 = keras.backend.max(out1, axis=0, keepdims=True)
      out1 = tf.squeeze(out1)
      out2 = tf.squeeze(self.avg_pooling2(f_list[1]), axis=[1, 2])
      out2 = keras.backend.max(out2, axis=0, keepdims=True)
      out2 = tf.squeeze(out2)
      arr1.append(out1)
      arr2.append(out2)
    out1 = tf.stack(arr1, axis=0)
    out2 = tf.stack(arr2, axis=0)
    out1 = tf.squeeze(self.fc1(self.d1(out1)))
    out2 = tf.squeeze(self.fc2(self.d2(out2)))
    out = tf.stack([out1, out2], axis = -1)
    out = tf.reshape(out, shape=(self.b_size, 2))
    return out


def MRNet_inc_model(combination = ["abnormal", "axial"]):
  b_size = 1
  model = keras.Sequential()
  model.add(MRNet_inception_layer(b_size))
  model(Input(shape=(None, 299, 299, 3)))
  model.summary()
  model.compile(optimizer='RMSprop', loss=tf.keras.losses.BinaryCrossentropy(), metrics=[binary_accuracy])

  checkpoint_path = "training_inception/" + combination[0] + "/" + combination[1]+"/cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
  return model, cp_callback