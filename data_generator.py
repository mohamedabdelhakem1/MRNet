import keras
import numpy as np
import os
from PIL import Image

class MRNet_data_generator(keras.utils.Sequence):
  def __init__(self, datapath, IDs, labels, batch_size = 32, shuffle=True,
               scale_to = (256, 256), label_type="abnormal", exam_type="axial",
               train=True):
    self.path = datapath
    self.IDs = IDs
    self.labels = labels
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.scale_to = scale_to
    self.label_type = label_type
    self.exam_type = exam_type
    if train:
      self.data_path = os.path.join(self.path, "train")
      self.data_type = "train"
    else:
      self.data_path = os.path.join(self.path, "valid")
      self.data_type = "valid"
    self.on_epoch_end()

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.IDs))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)
  
  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' 
    X = np.empty((self.batch_size), dtype=np.dtype)
    y = np.empty((self.batch_size), dtype=int)

    for i, ID in enumerate(list_IDs_temp):
        exam_path = os.path.join(self.data_path, self.exam_type)
        exam = np.load(os.path.join(exam_path, ID+'.npy'))
        e = []
        for s in exam:
          s = np.array(Image.fromarray(s).resize(self.scale_to))
          expanded = np.array([s, s, s])
          e.append(expanded)

        X[i] = np.array(e)
        y[i] = self.labels[ID][self.label_type]

    return X, y
  def __len__(self):
    'Denotes the number of batches per epoch'
    IDs_len = len(self.IDs[self.data_type][self.exam_type])
    return int(np.floor(IDs_len / self.batch_size))

  def __getitem__(self, index):
    'Generate one batch of data'
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    list_IDs_temp = [self.IDs[self.data_type][self.exam_type][k] for k in indexes]
    X, y = self.__data_generation(list_IDs_temp)
    return X, y