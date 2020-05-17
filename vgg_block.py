class VGG_block(keras.layers.Layer):
  def __init__(self):
    super(VGG_block, self).__init__()
    self.conv1_1 = Conv2D(input_shape=(3,224,224),filters=64,kernel_size=(3,3),
                          padding="same", activation="relu")




















    