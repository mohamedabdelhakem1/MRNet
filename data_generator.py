import keras
import numpy as np
import os
from PIL import Image

class MRNet_data_generator(keras.utils.Sequence):
  def __init__(self, datapath, IDs, labels, batch_size = 32, shuffle=True,
               scale_to = (256, 256), label_type="abnormal", exam_type="axial",
               train=True, model="vgg"):
    self.path = datapath
    self.n = 0
    self.IDs = IDs
    self.labels = labels
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.scale_to = scale_to
    self.label_type = label_type
    self.exam_type = exam_type
    self.model = model
    # self.cache_size = cache_size
    if train:
      self.data_path = os.path.join(self.path, "train")
      self.data_type = "train"
    else:
      self.data_path = os.path.join(self.path, "valid")
      self.data_type = "valid"
    # IDs_len = len(self.IDs[self.data_type][self.exam_type])
    # self.n_bachs = int(np.ceil(IDs_len / self.cache_size))
    # self.current = 0
    self.end = self.__len__()
    
    self.on_epoch_end()
    # print("initialized dg")

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.IDs[self.data_type][self.exam_type]))
    # self.current = 0
    # self._next_batch()
    if self.shuffle == True:
        np.random.shuffle(self.indexes)

  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' 
    # print("tototototototot")
    # print(list_IDs_temp)
    if self.model == "inception":
      y = np.empty((self.batch_size,1, 2), dtype=int)
    else:
      y = np.empty((self.batch_size), dtype=int)
    arr = []
    for i, ID in enumerate(list_IDs_temp):
        exam_path = os.path.join(self.data_path, self.exam_type)
        exam = np.load(os.path.join(exam_path, ID+'.npy'))
        e = []
        for s in exam:
          s = np.array(Image.fromarray(s).resize(self.scale_to), dtype=np.float32)
          expanded = np.array([s, s, s])
          e.append(expanded.reshape((self.scale_to[0], self.scale_to[1], 3)))

        e = np.array(e)
        arr.append(e)
        # X = np.stack(e, axis=0)
        _y = self.labels[ID][self.label_type]
        if self.model == "inception":
          y[i][0][0] = _y
          y[i][0][1] = _y
        else:
          y[i] = _y
        
        # y[i] = 155
    X = np.array(arr)
    # print(X.shape, y)
    return X, y
  def __len__(self):
    'Denotes the number of batches per epoch'
    IDs_len = len(self.IDs[self.data_type][self.exam_type])
    return int(np.floor(IDs_len / self.batch_size))

  def __getitem__(self, index):
    'Generate one batch of data'
    # if(index >= self.cache_size*self.current) or (index < self.cache_size*(self.current-1)):
    #   self.current = int(np.floor(index/self.cache_size))
    #   self._next_batch()
    #   return self.__getitem__(index)
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    # print(len(self.indexes))
    list_IDs_temp = [self.IDs[self.data_type][self.exam_type][k] for k in indexes]
    X, y = self.__data_generation(list_IDs_temp)
    # idx = index - (self.current-1) * self.cache_size
    # X = np.expand_dims(self.batch_x[idx], axis=0)
    # y = np.expand_dims(self.batch_y[idx], axis=0)
    return X, y

  # def _load_batch(self, index):
  #   mx = (index+1)*self.cache_size
  #   if(mx >= self.__len__()):
  #     indexes = self.indexes[index*self.cache_size:]
  #   else:
  #     indexes = self.indexes[index*self.cache_size:mx]
  #   list_IDs_temp = [self.IDs[self.data_type][self.exam_type][k] for k in indexes]
  #   X, y = self.__data_generation(list_IDs_temp)
  #   return X, y

  # def _next_batch(self):
  #   if self.current >= self.n_bachs:
  #     self.current = 0
  #   self.batch_x = None
  #   del self.batch_x
  #   self.batch_x, self.batch_y = self._load_batch(self.current)
  #   self.current += 1

  def __next__(self):
    # print("toototot")
    if self.n >= self.end:
      self.n = 0
    result = self.__getitem__(self.n)
    self.n += 1
    return result
