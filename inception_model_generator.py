import keras
import tensorflow as tf
import numpy as np 
from keras.regularizers import l2
from keras import Model
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input, concatenate, GlobalAveragePooling2D, AveragePooling2D, Flatten
import os


class inceptionV3():
  def __init__(self,shape ,weightsPath = None):
    self.shape =shape;
    self.weightsPath = weightsPath;
    
  def inceptionModlueA(self ,inp,filter1_1x1,filter2_pool,filter3_1x1,filter3_3x3,filter4_1x1,filter4_3x3,filter4_3x3_2,kernel_init="glorot_uniform",
    bias_init="zeros",kernel_regularizer=l2(0.0002),name=None):
    conv1  = Conv2D(filters=filter1_1x1,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init
    ,kernel_regularizer=kernel_regularizer)(inp)
  
    conv2_1  = AveragePooling2D((3,3),strides=(1,1),padding='same')(inp)
    conv2_2 =  Conv2D(filters=filter2_pool,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(conv2_1)
    
    conv3_1 = Conv2D(filters=filter3_1x1,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(inp)
    conv3_2 = Conv2D(filters=filter3_3x3,kernel_size=(5,5),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(conv3_1)
    
    conv4_1 = Conv2D(filters=filter4_1x1 , kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(inp)
    conv4_2 = Conv2D(filters= filter4_3x3 ,kernel_size=(3,3),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(conv4_1)
    conv4_3 = Conv2D(filters= filter4_3x3_2 ,kernel_size=(3,3),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(conv4_2)
    
    output = concatenate([conv1, conv2_2, conv3_2, conv4_3], axis=3, name=name)
    return output

  def inceptionModuleB(self,inp,filter1_1x1,filter2_pool,filter3_1x1,filter3_1xn,filter3_nx1,filter4_1x1,filter4_1xn,filter4_nx1,filter4_1xn_1,filter4_nx1_2,kernel_init="glorot_uniform",
    bias_init="zeros",name=None,kernel_regularizer=l2(0.0002)):
    conv1  = Conv2D(filters=filter1_1x1,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(inp)

    conv2_1  = AveragePooling2D((3,3),strides=(1,1),padding='same')(inp)
    conv2_2 =  Conv2D(filters=filter2_pool,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(conv2_1)
    
    conv3_1 = Conv2D(filters=filter3_1x1,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(inp)
    conv3_2 = Conv2D(filters=filter3_1xn,kernel_size=(1,7),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(conv3_1)
    conv3_3 = Conv2D(filters=filter3_nx1,kernel_size=(7,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(conv3_2)
    
    conv4_1 = Conv2D(filters=filter4_1x1,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(inp)
    conv4_2 = Conv2D(filters=filter4_1xn,kernel_size=(1,7),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(conv4_1)
    conv4_3 = Conv2D(filters=filter4_nx1,kernel_size=(7,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(conv4_2)
    conv4_4 = Conv2D(filters=filter4_1xn_1,kernel_size=(1,7),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(conv4_3)
    conv4_5 = Conv2D(filters=filter4_nx1_2,kernel_size=(7,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(conv4_4)
    
    output = concatenate([conv1, conv2_2, conv3_3, conv4_5], axis=3, name=name)
    return output
    
  def inceptionModuleC(self,inp,filter1_1x1,filter2_1x1,filter3_1x1,filter3_1x3,filter3_3x1,filter4_1x1,filter4_3x3,filter4_1x3,filter4_3x1,kernel_init="glorot_uniform",
    bias_init="zeros",name=None,kernel_regularizer=l2(0.0002)):
    conv1  = Conv2D(filters=filter1_1x1,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init)(inp)

    conv2_1  = AveragePooling2D((3,3),strides=(1,1),padding='same')(inp)
    conv2_2 =  Conv2D(filters=filter2_1x1,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(conv2_1)
    
    conv3_1 = Conv2D(filters=filter3_1x1,kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(inp)
    conv3_2 = Conv2D(filters=filter3_1x3,kernel_size=(1,3),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(conv3_1)
    conv3_3 = Conv2D(filters=filter3_3x1,kernel_size=(3,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(conv3_1)
    conv_3 = concatenate([conv3_2,conv3_3],axis=3);
    
    conv4_1 = Conv2D(filters=filter4_1x1 , kernel_size=(1,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(inp)
    conv4_2 = Conv2D(filters= filter4_3x3 ,kernel_size=(3,3),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(conv4_1)
    conv4_3 = Conv2D(filters= filter4_1x3 ,kernel_size=(1,3),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(conv4_2)
    conv4_4 = Conv2D(filters= filter4_3x1 ,kernel_size=(3,1),padding='same',activation='relu',bias_initializer= bias_init,kernel_initializer=kernel_init,kernel_regularizer=kernel_regularizer)(conv4_2)
    conv_4 = concatenate([conv4_3,conv4_4],axis=3);

    output = concatenate([conv1, conv2_2,conv_3,conv_4], axis=3, name=name)
    return output

  def inceptionModlueD(self,x,filter1_3x3 , filter2_1x1, filter2_3x3 ,  filter2_3x3_2, name=None,kernel_regularizer=l2(0.0002)):
    conv1 = Conv2D(filters=filter1_3x3, kernel_size=(3,3),strides=(2,2),padding='valid',activation='relu',kernel_regularizer=kernel_regularizer)(x);

    conv2_1 = Conv2D(filters=filter2_1x1, kernel_size=(1,1),padding='same',activation='relu',kernel_regularizer=kernel_regularizer)(x);
    conv2_2 = Conv2D(filters=filter2_3x3, kernel_size=(3,3),padding='same',activation='relu',kernel_regularizer=kernel_regularizer)(conv2_1);
    conv2_3 = Conv2D(filters=filter2_3x3_2, kernel_size=(3,3),strides=(2,2),padding='valid',activation='relu',kernel_regularizer=kernel_regularizer)(conv2_2);

    conv3 = MaxPool2D(pool_size=(3,3) , strides=(2,2),padding='valid')(x);

    output = concatenate([conv1, conv2_3,conv3], axis=3, name=name)
    return output
  def inceptionModlueE(self,x,filter1_1x1,filter1_3x3 , filter2_1x1, filter2_1x7 ,filter2_7x1,filter2_3x3,name=None,kernel_regularizer=l2(0.0002)):
    conv1_1 = Conv2D(filters=filter1_1x1,kernel_size=(1,1),padding='same',activation='relu',kernel_regularizer=kernel_regularizer)(x);
    conv1_2 = Conv2D(filters=filter1_3x3,kernel_size=(3,3),strides=(2,2),padding='valid',activation='relu',kernel_regularizer=kernel_regularizer)(conv1_1);
    
    conv2_1 =  Conv2D(filters=filter2_1x1,kernel_size=(1,1),padding='same',activation='relu',kernel_regularizer=kernel_regularizer)(x);
    conv2_2 =  Conv2D(filters=filter2_1x7,kernel_size=(1,7),padding='same',activation='relu',kernel_regularizer=kernel_regularizer)(conv2_1);
    conv2_3 =  Conv2D(filters=filter2_7x1,kernel_size=(7,1),padding='same',activation='relu',kernel_regularizer=kernel_regularizer)(conv2_2);
    conv2_4 =  Conv2D(filters=filter2_3x3,kernel_size=(3,3),strides=(2,2),padding='valid',activation='relu',kernel_regularizer=kernel_regularizer)(conv2_3);
    
    conv3 = MaxPool2D(pool_size=(3,3),strides=2,padding='valid')(x);
    output = concatenate([conv1_2, conv2_4,conv3], axis=3, name=name)
    return output
  def getModel(self):
    input = Input(shape=self.shape);
    # 224x224x3
    x = Conv2D(filters=32,kernel_size=(3,3),strides=(2,2),padding='valid',activation='relu',kernel_regularizer=l2(0.0002))(input)
    # 112x112x32
    x = Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',kernel_regularizer=l2(0.0002))(x)
    # 110x110x32
    x = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',kernel_regularizer=l2(0.0002))(x)  
    # 110x110x64
    x = MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
    # 55x55x64
    x = Conv2D(filters=80,kernel_size=(1,1),strides=(1,1),padding='valid',activation='relu',kernel_regularizer=l2(0.0002))(x)
    # 55x55x80
    x = Conv2D(filters=192,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',kernel_regularizer=l2(0.0002))(x)
    # 53x53x192
    x = MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
    # 26x26x192
    x = self.inceptionModlueA(x,filter1_1x1=64,filter2_pool=32,filter3_1x1=48,filter3_3x3=64,filter4_1x1=64,filter4_3x3=96,filter4_3x3_2=96);
    # 26x26x256
    x = self.inceptionModlueA(x,filter1_1x1=64,filter2_pool=64,filter3_1x1=48,filter3_3x3=64,filter4_1x1=64,filter4_3x3=96,filter4_3x3_2=96);
    # 26x26x288
    x = self.inceptionModlueA(x,filter1_1x1=64,filter2_pool=64,filter3_1x1=48,filter3_3x3=64,filter4_1x1=64,filter4_3x3=96,filter4_3x3_2=96);
    # 26x26x288

    x = self.inceptionModlueD(x,filter1_3x3=384 , filter2_1x1=64, filter2_3x3=96 ,  filter2_3x3_2=96);
    # 13x13x768

    x =  self.inceptionModuleB(x,filter1_1x1=192,filter2_pool=192,filter3_1x1=128,filter3_1xn=128,filter3_nx1=192,filter4_1x1=128,filter4_1xn=128,filter4_nx1=128
                          ,filter4_1xn_1=128,filter4_nx1_2=192);
    x =  self.inceptionModuleB(x,filter1_1x1=192,filter2_pool=192,filter3_1x1=128,filter3_1xn=128,filter3_nx1=192,filter4_1x1=128,filter4_1xn=128,filter4_nx1=128
                          ,filter4_1xn_1=128,filter4_nx1_2=192);
    x =  self.inceptionModuleB(x,filter1_1x1=192,filter2_pool=192,filter3_1x1=160,filter3_1xn=160,filter3_nx1=192,filter4_1x1=160,filter4_1xn=160,filter4_nx1=160
                          ,filter4_1xn_1=160,filter4_nx1_2=192);
                          
    x =  self.inceptionModuleB(x,filter1_1x1=192,filter2_pool=192,filter3_1x1=160,filter3_1xn=160,filter3_nx1=192,filter4_1x1=160,filter4_1xn=160,filter4_nx1=160
                          ,filter4_1xn_1=160,filter4_nx1_2=192);

    x =  self.inceptionModuleB(x,filter1_1x1=192,filter2_pool=192,filter3_1x1=192,filter3_1xn=192,filter3_nx1=192,filter4_1x1=192,filter4_1xn=192,filter4_nx1=192
                          ,filter4_1xn_1=192,filter4_nx1_2=192);
    # x1 = x;
    # print(x1.shape)
    # x1 = AveragePooling2D((5, 5), strides=3,padding='valid')(x1)
    # print(x1.shape)
    # x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    # print(x1.shape)
    # x1 = Flatten()(x1)
    # x1 = Dense(1024, activation='relu')(x1)
    # x1 = Dropout(0.7)(x1)
    # x1 = Dense(1, activation='sigmoid', name='auxilliary_output_1')(x1)


    x = self.inceptionModlueE(x,filter1_1x1=192,filter1_3x3=320 , filter2_1x1=192, filter2_1x7=192 ,filter2_7x1 =192 ,filter2_3x3=192);

    x = self.inceptionModuleC(x,filter1_1x1=320,filter2_1x1=192,filter3_1x1=384,filter3_1x3=384,filter3_3x1=384,filter4_1x1=448,filter4_3x3=384,filter4_1x3=384,filter4_3x1=384);

    x = self.inceptionModuleC(x,filter1_1x1=320,filter2_1x1=192,filter3_1x1=384,filter3_1x3=384,filter3_3x1=384,filter4_1x1=448,filter4_3x3=384,filter4_1x3=384,filter4_3x1=384);
    # print(x.shape)
    # x = GlobalAveragePooling2D()(x)
    # print(x.shape)
    # x = Dropout(0.4)(x)
    # x = Dense(1, activation='sigmoid', name='output')(x)
    
    # auxillary
    # model = Model(inputs=input, outputs =[x, x1], name='inception_v3')
    
    model = Model(inputs=input, outputs =x, name='inception_v3')
    # model.summary()
    
    if self.weightsPath:
        model.load_weights(self.weightsPath)
    return model

class MRNet_inception_layer(keras.layers.Layer):
  def __init__(self, batch_size):
    super(MRNet_inception_layer, self).__init__()
    self.inception = inceptionV3((299, 299, 3)).getModel()
    self.avg_pooling1 = AveragePooling2D(pool_size=(8, 8), padding="same")
    self.d1 = Dropout(0.5)
   
    self.fc1 = Dense(1, activation="sigmoid", input_dim=2048)
    self.b_size = batch_size

  def compute_output_shape(self, input_shape):
    return (None, 1)

  def call(self, inputs):
    arr1 = []
    for index in range(self.b_size):
      f_list = self.inception(inputs[index])
      out1 = tf.squeeze(self.avg_pooling1(f_list), axis=[1, 2])
      out1 = keras.backend.max(out1, axis=0, keepdims=True)
      out1 = tf.squeeze(out1)
      arr1.append(out1)
    out1 = tf.stack(arr1, axis=0)
    out1 = self.fc1(self.d1(out1))
    return out1


def MRNet_inc_model(combination = ["abnormal", "axial"]):
  b_size = 1
  model = keras.Sequential()
  model.add(MRNet_inception_layer(b_size))
  model(Input(shape=(None, 299, 299, 3)))
  model.summary()
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
  initial_learning_rate = 0.045
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=2260,
    decay_rate=0.94,
    staircase=True)
  model.compile(optimizer=tf.keras.optimizers.RMSprop(
    learning_rate=lr_schedule,
    rho=0.9,
    momentum=0.9,
    epsilon=1,
    clipnorm=2.0
  ),   loss=keras.losses.BinaryCrossentropy(),
      metrics=METRICS)
    
  checkpoint_path = "training_inception/" + combination[0] + "/" + combination[1]+"/"
  checkpoint_dir = os.path.dirname(checkpoint_path)
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  os.chdir(checkpoint_dir)
  cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_dir+"weights{epoch:02d}.hdf5",
                                                 save_weights_only=True,
                                                 verbose=1)
  # print(os.path.abspath(os.getcwd()))
  # checkpoint_path = "training_inception/" + combination[0] + "/" + combination[1]+"/cp.ckpt"
  # checkpoint_dir = os.path.dirname(checkpoint_path)
  # print(checkpoint_dir)
  # if not os.path.exists(checkpoint_dir):
  #   os.makedirs(checkpoint_dir)
  # cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
  #                                                save_weights_only=True,
  #                                                verbose=1)
  return model, cp_callback
  # uncomment for adding the auxillary classifier
# class MRNet_inception_layer(keras.layers.Layer):
#   def __init__(self, batch_size):
#     super(MRNet_inception_layer, self).__init__()
#     self.inception = inceptionV3((299, 299, 3)).getModel()
#     self.avg_pooling1 = AveragePooling2D(pool_size=(8, 8), padding="same")
#     self.avg_pooling2 = AveragePooling2D(pool_size=(5, 5), padding="same")
#     self.d1 = Dropout(0.5)
#     self.d2 = Dropout(0.5)

#     self.fc1 = Dense(1, activation="sigmoid", input_dim=2048)
#     self.fc2 = Dense(1, activation="sigmoid", input_dim=128)
#     self.b_size = batch_size

#   def compute_output_shape(self, input_shape):
#     return (None, 2)

#   def call(self, inputs):
#     arr1 = []
#     arr2 = []
#     for index in range(self.b_size):
#       f_list = self.inception(inputs[index])
#       out1 = tf.squeeze(self.avg_pooling1(f_list[0]), axis=[1, 2])
#       out1 = keras.backend.max(out1, axis=0, keepdims=True)
#       out1 = tf.squeeze(out1)
#       out2 = tf.squeeze(self.avg_pooling2(f_list[1]), axis=[1, 2])
#       out2 = keras.backend.max(out2, axis=0, keepdims=True)
#       out2 = tf.squeeze(out2)
#       arr1.append(out1)
#       arr2.append(out2)
#     out1 = tf.stack(arr1, axis=0)
#     out2 = tf.stack(arr2, axis=0)
#     out1 = tf.squeeze(self.fc1(self.d1(out1)))
#     out2 = tf.squeeze(self.fc2(self.d2(out2)))
#     out = tf.stack([out1, out2], axis = -1)
#     out = tf.reshape(out, shape=(self.b_size, 2))
#     return out
