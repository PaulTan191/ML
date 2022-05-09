import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    
    """
    A class to contain a simple tensor flow LSTM model. Inherits from tf.keras.Model class.
    
    In the future, this class should be expanded to contain a model which takes in multiple 
    inputs, and uses a more complex architecture.
    
    
    ...
    Attributes
    ----------
    inp_dim : integer
        Dimension of the feature space
    out_dim : integer
        Dimension of the output space
    theta : [int]
        Hyperparameters of the LSTM model; number of units in each hidden layer. The length of theta is the number of hidden layers. Where hidden layers here excludes the output layer. 
    
    activation : tf.keras.activations object
        Activation function for the hidden layers (not the final activation function)
    """
    def __init__(self, theta = [64]):
        super(Model, self).__init__()
        self.theta = theta
        
        self.hidden_layers = []
        
        for i in range(len(theta)):
            self.hidden_layers.append(tf.keras.layers.LSTM(theta[i],return_sequences = True))
        
        self.output_layer = tf.keras.layers.Dense(units = 1)

    
    def call(self, inputs):
        x = self.hidden_layers[0](inputs)
        for i in range(len(self.hidden_layers)-1):
            x = self.hidden_layers[i](x)
        return self.output_layer(x)
    

    
                 
                 
                 
                 
                 
                 
                 
                 
                 