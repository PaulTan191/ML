import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import Model

## TODO, implement a general LSTM model , using windows.

class Stock:
    
    """
    A class to represent a Stock timeseries with a randomly initizalied LSTM models outputting multiple predictions.
    
    ...
    
    Attributes
    ----------
    history : np.array
        Time series list of past stock prices
    N : int
        Number of random instantiations of initial weights.
        Multiple models are trained as a form of validation, to understand how robust the predictions are.
    lookahead : int
        Number of time steps into the future to predict
    epochs : int
        Number of training epochs per model.
    predictions : np.array of shape (N,lookahead + len(history))
        Model prediction of history time series plus lookahead steps into the future
    losses : np.array of shape (N, epochs)
        Training Losses
    stockname : String
        Name of stock
    theta : [int]
        Hyperparameters of the LSTM model; number of units in each hidden layer. The length of theta is the number of hidden layers. Where hidden layers here excludes the output layer. 
    
    Methods
    -------
    run(self):
        Trains multiple models and makes stock predictions.
    plot_losses(self,ax = None, title = "Loss",xlabel= "Epochs",ylabel="MSE",in_figsize = (5,5)):
        Plots the loss functions
        
    
    
    
    """
    def __init__(self, history, N, stockname, lookahead = 1, epochs = 1000,theta = [64], lookbacks = 5):


      # Require lookbacks < len(history)
        self.predictions = np.zeros((lookbacks,N,len(history)+lookahead))
        self.losses = np.zeros((lookbacks,N,epochs))

        
        self.input = np.linspace(0,len(history),len(history)).reshape(1,len(history),1)
        # Standardizing the price, for training purposes.
        self.history = ((history - history.mean())/history.std()).reshape(1,len(history),1)
        self.epochs = epochs
        
        self.lookahead = lookahead
        self.N = N
        self.stockname = stockname
        
        lookbacklst = range(lookbacks)
        self.histories = [self.history[:,i:,:] for i in lookbacklst]

        
        
    # TODO, possibly make subset lookback validation optional for interface clarity. 
    def run(self):
        for i in range(len(self.histories)):
            for j in range(self.N):
                model = Model.Model(theta = [256])
                
                model.compile(loss =tf.keras.losses.MeanSquaredError(),optimizer = tf.keras.optimizers.Adam())

                
                train = model.fit(self.input[:,i:,:]-i,self.histories[i],epochs = self.epochs, verbose = 0)
                
                n =  len(self.histories[i][0,:,0])+self.lookahead
                
                future = np.linspace(0,n,n).reshape(1,n,1)
                self.predictions[i,j,i:] = model.predict(future)[0,:,0]
                self.losses[i,j] = train.history['loss']
            
            
    
    def plot_losses(self,ax = None, title = "Loss",xlabel= "Epochs",ylabel="MSE",in_figsize = (5,5),log = True):
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=in_figsize)
        for i in range(len(self.histories)):
          for j in range(self.N):
            ax.plot(self.losses[i,j])
        if log:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        
    # TODO, add clear labels and informative colouring.
    def plot_predictions(self,ax = None, title = "Model Predictions: ",xlabel= "Timestep",ylabel="Standardized Price",in_figsize = (5,5)):
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=in_figsize)

        for i in range(len(self.histories)):
          for j in range(self.N):
            ax.plot(self.predictions[i,j][:-self.lookahead],'-k')
            ax.plot(np.linspace(len(self.history[0,:,0])-1,len(self.history[0,:,0]),2),self.predictions[i,j][-self.lookahead-1:])
        ax.plot(self.history[0,:,0],'r')
        ax.set_title(title+ self.stockname)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
