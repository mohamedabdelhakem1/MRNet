import numpy as np
import keras
import os
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import np_utils
from keras.utils import plot_model
import scipy.misc
from keras.initializers import glorot_uniform
from matplotlib.pyplot import imshow
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal as RN
from tensorflow.keras.initializers import GlorotNormal as GN
from tensorflow.keras.initializers import GlorotUniform as GU

K.set_image_data_format('channels_last')
K.set_learning_phase(1)
class ResNet50():

  def __init__(self,shape ,weightsPath = None):
    self.shape =shape;
    self.weightsPath = weightsPath;

  def identity_block(self,X,f , filters, stage, block):
    """
    Implementation of the identity block
    
    Arguments:
    X -- input tensor of shape (m, #heddin layers in prev, #W in prev, #channels in prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (#H, #W, #C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    
    # Second component of main path 
    X = Conv2D(filters = F2, kernel_size=(f,f), strides = (1,1), padding='same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(filters = F3, kernel_size=(1,1), strides = (1,1), padding="valid", name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    
    return X

  def convolutional_block(self,X, filters, stage, block, s =2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    #F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(filters = filters[0], kernel_size= (1, 1), strides = (s,s),padding="valid", name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    

    # Second component of main path 
    X = Conv2D(filters = filters[1], kernel_size=(3,3), strides=(1,1), name = conv_name_base + '2b', padding="same",kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name= bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(filters = filters[2], kernel_size=(1,1), strides = (1,1), name= conv_name_base + '2c',padding="valid", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(filters = filters[2], kernel_size= (1,1), strides=(s,s), name=conv_name_base + '1', padding="valid", kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base+'1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X_shortcut,X])
    X = Activation("relu")(X)
    
    
    return X

  def getModel(self,classes = 1):
    """
    Arguments:
    classes -- integer, number of classes
    Returns:
    model -- a Model() instance in Keras
    """
    input = Input(shape=self.shape);
    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    st2filters=[64, 64, 256]
    X = self.convolutional_block(X,st2filters, 2, 'a', 1)
    X = self.identity_block(X, 3, [64, 64, 256], 2, 'b')
    X = self.identity_block(X, 3, [64, 64, 256], 2, 'c')


    # Stage 3
    X = self.convolutional_block(X, [128,128,512],3,'a', 2)
    X = self.identity_block(X, 3, [128,128,512], 3, 'b')
    X = self.identity_block(X, 3, [128,128,512], 3, 'c')
    X = self.identity_block(X, 3, [128,128,512], 3, 'd')

    # Stage 4 
    X = self.convolutional_block(X, [256,256,1024],4 , 'a', 2)
    X = self.identity_block(X, 3, [256, 256, 1024],4 , 'b')
    X = self.identity_block(X, 3, [256, 256, 1024],4 , 'c')
    X = self.identity_block(X, 3, [256, 256, 1024],4 , 'd')
    X = self.identity_block(X, 3, [256, 256, 1024],4 , 'e')
    X = self.identity_block(X, 3, [256, 256, 1024],4 , 'f')

    # Stage 5 
    X = self.convolutional_block(X, [256,256,2048], 5,'a', 3)
    X = self.identity_block(X, 3, [256,256,2048], 5, 'b')
    X = self.identity_block(X, 3, [256,256,2048], 5, 'c')

    # AVGPOOL (Ã¢â€°Ë†1 line). Use "X = AveragePooling2D(...)(X)"
    #X = AveragePooling2D((2,2), name='avg_pool')(X)
    

    # output layer
    #X = Flatten()(X)
    #X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = input, outputs = X, name='ResNet50')

    if self.weightsPath:
        model.load_weights(self.weightsPath)

    return model

class MRNet_ResNet_layer(keras.layers.Layer):
    
  def __init__(self, input_shape,batch_size):
    super(MRNet_ResNet_layer, self).__init__()
    self.ResNet = ResNet50(input_shape[2:]).getModel()
    self.avg_pooling = AveragePooling2D((5, 5), name='avg_pool', padding="same")
    self.fc = Dense(1, activation='sigmoid', name='fc' + str(1), kernel_initializer = glorot_uniform(seed=0))
    self.b_size = batch_size

  def compute_output_shape(self, input_shape):
    return (None, 2)

  @tf.function
  def call(self, inputs):
    arr = []
    for index in range(self.b_size):
      out = self.ResNet(inputs[index])
      out = tf.squeeze(self.avg_pooling(out), axis=[1, 2])
      out = keras.backend.max(out, axis=0, keepdims=True)
      out = tf.squeeze(out)
      arr.append(out)
    output = tf.stack(arr, axis=0)
    output = self.fc(output)
    return output

  def compute_output_shape(self, input_shape):
    return (None, 1)


def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))

def MRNet_ResNet_model(batch_size,lr, combination = ["abnormal", "axial"]):
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
  model.add(MRNet_ResNet_layer((None, None, 224, 224, 3), b_size))
  model(Input(shape=(None, 224, 224, 3)))
  #print(model(tf.ones((12,51, 224, 224, 3))).shape)
  model.compile(
      #optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=1e-2),
      optimizer=tf.keras.optimizers.Adam(lr=lr, decay=1e-2),
      loss=keras.losses.BinaryCrossentropy(),metrics=METRICS)
  data_path = "/content/gdrive/My Drive/Colab Notebooks/MRNet/"
  checkpoint_dir = data_path+"training_ResNet/" + combination[0] + "/" + combination[1] + "/"
  # checkpoint_dir = os.path.dirname(checkpoint_path)
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  os.chdir(checkpoint_dir)
  cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_dir+"weights.{epoch:02d}.hdf5",
                                                 save_weights_only=True,
                                                 verbose=1)
  tcb = TestCallback(model)
  return model, [cp_callback, tcb]

class TestCallback(tf.keras.callbacks.Callback):
  def __init__(self, model):
    super(TestCallback, self).__init__()
    self.model = model

  def on_epoch_end(self, epoch, logs=None):
    if(epoch == 0):
      self.w = self.model.layers[0].get_weights()[0]
      return
    self.w_after = self.model.layers[0].get_weights()[0]
    print('  TestCallback: ', (self.w == self.w_after).all())
    self.w = self.w_after