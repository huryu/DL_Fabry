#This class should be externalized!!!
# Originated from "https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html"

# If you want to input multiple data (multi_input_model), adjust the output of "def generate"...
# That function should yield

import numpy as np

class DataGenerator(object):
    '''Generates data for Keras'''
    
    def __init__(self, vert_size = 192, hori_size = 256, color_chn = 3, batch_size = 32, shuffle = True):
        '''Initialization'''
        self.vert_size = vert_size
        self.hori_size = hori_size
        self.color_chn = color_chn
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, labels, data_dict, list_IDs):
        '''Generates batches of samples'''
        
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)
            # Generate batches
            imax = int(len(indexes)/self.batch_size)
        
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
            
                # Generate data
                X, y = self.__data_generation(labels, data_dict, list_IDs_temp)
            
                yield X, y

    def __get_exploration_order(self, list_IDs):
        '''Generates order of exploration'''
        # Find exploration order
    
        indexes = np.arange(len(list_IDs))
    
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, labels, data_dict, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.vert_size, self.hori_size, self.color_chn))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, :, :, :] = np.load(data_dict[ID])

            # Store class
            y[i] = labels[ID]

        return X, y
