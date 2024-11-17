from abc import ABCMeta
import tensorflow as tf
import numpy as np

from cms.modules.parameter_manager import DatasetParam
from cms.modules.augmet_manager import train_augment, test_augment

class BaseTrainDataloader(metaclass=ABCMeta):

    def __init__(self, config:DatasetParam,num_train=50000, drop_label=True):
        self.config = config
        self.load_dataset(num_train,drop_label)
        
    def load_dataset(self, num_train, drop_label ,max_pix=255):
        if self.config.dataset == "cifar100":
            x,y =  tf.keras.datasets.cifar100.load_data(label_mode="fine")
            x_train, y_train = x[0][:num_train], x[1][:num_train]
            # erase train label for semi-supervised
            unlabeled_count = int(self.config.unlabeled_ratio*y_train.shape[0])
            rng = np.random.default_rng(seed=123)
            unlabeled_index_array = rng.permutation(np.arange(y_train.shape[0]))[:unlabeled_count]
            if drop_label:
                y_train[unlabeled_index_array] = -1
            x_test, y_test = y[0], y[1]
            x_train, x_test = tf.convert_to_tensor(x_train/max_pix, dtype=tf.float32), tf.convert_to_tensor(x_test/max_pix, dtype=tf.float32)
            y_train, y_test = tf.convert_to_tensor(y_train, dtype=tf.uint8), tf.convert_to_tensor(y_test, dtype=tf.uint8)
            self.train_dataset = (x_train, y_train)
            self.test_dataset  = (x_test, y_test)
            
    @property
    def epoch_iteration(self):
        return int(self.train_dataset[0].shape[0]/self.config.batch_size)
    
class CMSDataloader(BaseTrainDataloader):
    def make_train_dataset(self, is_train_augmentation):
        train_ds = tf.data.Dataset.from_tensor_slices(self.train_dataset)
        train_ds = train_ds.shuffle(train_ds.cardinality())
        map_function = self._augmentation(is_train_augmentation=is_train_augmentation)
        train_ds = train_ds.map(map_function, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        train_ds = train_ds.batch(self.config.batch_size,drop_remainder=True)
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        train_ds = train_ds.repeat()

        return train_ds
    
    def make_memory_dataset(self, is_train_augmentation):
        memory_ds = tf.data.Dataset.from_tensor_slices(self.train_dataset)
        memory_ds = memory_ds.shuffle(memory_ds.cardinality())
        map_function = self._augmentation(is_train_augmentation=is_train_augmentation)
        memory_ds = memory_ds.map(map_function, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        memory_ds = memory_ds.batch(self.config.batch_size,drop_remainder=False)
        memory_ds = memory_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return memory_ds
    
    def make_test_dataset(self, test_batch_size):
        test_ds = tf.data.Dataset.from_tensor_slices(self.test_dataset)
        test_ds = test_ds.shuffle(test_ds.cardinality())
        map_function = self._augmentation(is_train_augmentation=False)
        test_ds = test_ds.map(map_function, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        test_ds = test_ds.batch(test_batch_size, drop_remainder=False)

        return test_ds
    
    def _augmentation(self, is_train_augmentation):
        def _map_function(x:tf.TensorArray,y:tf.TensorArray):
            x1 = train_augment(x, self.config)
            x2 = train_augment(x, self.config) 
            return x1, x2, y
        
        def _map_test_function(x:tf.TensorArray,y:tf.TensorArray):
            x1 = test_augment(x, self.config)
            x2 = test_augment(x, self.config)
            return x1, x2, y
        
        if is_train_augmentation:
            return _map_function
        else:
            return _map_test_function
    