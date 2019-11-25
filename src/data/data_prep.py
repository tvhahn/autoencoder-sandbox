import numpy as np
from sklearn.utils import shuffle
# import random

class DataPrep:
    def __init__(self, keras_dataset, class_normal, random_int=16, print_shapes=False):
        
        self.class_normal = class_normal
        self.random_int = random_int
        self.print_shapes = print_shapes

        (X_train_all, y_train_all), (X_test, y_test) = keras_dataset.load_data()

        # shuffle the data, as a precaution
        self.X_train_all, self.y_train_all = shuffle(X_train_all, y_train_all, random_state=random_int)
        self.X_test, self.y_test = shuffle(X_test, y_test, random_state=random_int)

        # convert all to dtype float32
        self.X_train_all = self.X_train_all.astype('float32')
        self.X_test = self.X_test.astype('float32')

    def remove_classes(self, X_val_slim, y_val_slim):
        """Funciton to remove classes from train/val set"""
        
        class_to_remove = [i for i in range(0,10)]
        class_to_remove.remove(self.class_normal)
        class_to_remove = np.array(class_to_remove,dtype='uint8')

        # start with y_valid_slim
        index_to_delete = []
        for i, class_digit in enumerate(y_val_slim):
            if class_digit in class_to_remove:
                index_to_delete.append(i)

        y_val_slim = np.delete(y_val_slim, index_to_delete)
        X_val_slim = np.delete(X_val_slim, index_to_delete, axis=0)

        return X_val_slim, y_val_slim
        
    def train_test_split(self):
        # split the data between train and validation sets, and scale
        # also have a "slimmed down" data set that has only positive classes that will be used to train the autoencoder
        self.X_val, self.X_val_slim, self.X_train = self.X_train_all[:5000] / 255.0, \
                                                self.X_train_all[5000:10000] / 255.0, \
                                                self.X_train_all[10000:] / 255.0
        self.y_val, self.y_val_slim, self.y_train = self.y_train_all[:5000],self.y_train_all[5000:10000], self.y_train_all[10000:]

        # also scale the X_test
        self.X_test = self.X_test / 255.0
        
        self.X_train_slim, self.y_train_slim = self.remove_classes(self.X_train, self.y_train)
        self.X_val_slim, self.y_val_slim = self.remove_classes(self.X_val_slim, self.y_val_slim)
        
        if self.print_shapes == True:
            print('X_val shape:', self.X_val.shape)
            print('y_val shape:', self.y_val.shape)
            print('X_val_slim shape:', self.X_val_slim.shape)
            print('y_val_slim shape:', self.y_val_slim.shape)
            print('X_train shape:', self.X_train.shape)
            print('y_train shape:', self.y_train.shape)
            print('X_train_slim shape:', self.X_train_slim.shape)
            print('y_train_slim shape:', self.y_train_slim.shape)
        
        return (self.X_train, 
                self.y_train, 
                self.X_train_slim, 
                self.y_train_slim, 
                self.X_val,
                self.y_val,
                self.X_val_slim,
                self.y_val_slim,
                self.X_test,
                self.y_test)