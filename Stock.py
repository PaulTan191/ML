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
    histories : [[int]] 
        List of N history stock price histories of size (lookbacks, f(lookbacks)). Where f(lookbacks) = lookbacks * lookback_interval indicates that the lists in the array are of variable size.
    
    Methods
    -------
    run(self):
        Trains multiple models and makes stock predictions.
    plot_losses(self,ax = None, title = "Loss",xlabel= "Epochs",ylabel="MSE",in_figsize = (5,5)):
        Plots the loss functions
    plot_predictions(self,ax = None, title = "Model Predictions: ",xlabel= "Timestep",ylabel="Standardized Price",in_figsize = (5,5)):
        Plots the predictions of the models for each weight instantiation and for each specified training subsets of the historical price.
    
    
    
    """
    def __init__(self, history, N, stockname, lookahead = 50, epochs = 1000,theta = [64], lookbacks = 5, lookback_interval = 10,batch_size = 50):


      # Require lookbacks < len(history)
        self.batch_size = batch_size
        self.predictions = np.zeros((lookbacks,N,len(history)+lookahead))
        self.losses = np.zeros((lookbacks,N,epochs))

        
        self.input = np.linspace(0,len(history),len(history)).reshape(1,len(history),1)
        # Standardizing the price, for training purposes.
        self.history = ((history - history.mean())/history.std()).reshape(1,len(history),1)
        self.epochs = epochs
        
        self.lookahead = lookahead
        self.N = N
        self.stockname = stockname
        self.lookback_interval = lookback_interval
        self.histories = [self.history[:,self.lookback_interval * i:,:] for i in range(lookbacks)]

        
        
    # TODO, implement a validation option which chooses to look at subsets of the stock price history.
    # e.g, predicts models looking back {15,14,13,12,11,10} time steps in the past. 
    # Motivation for this loosely said: we do not know how far back the relevent data is   
    def run(self):
        counter = 0
        for i in range(len(self.histories)):
            for j in range(self.N):
                counter += 1
                model = Model.Model(theta = [256])
                
                model.compile(loss =tf.keras.losses.MeanSquaredError(),optimizer = tf.keras.optimizers.Adam())

                
                train = model.fit(self.input[:,self.lookback_interval*i:,:]-i*self.lookback_interval,self.histories[i],epochs = self.epochs, batch_size = self.batch_size,verbose = 0)
                
                n =  len(self.histories[i][0,:,0])+self.lookahead
                
                future = np.linspace(0,n,n).reshape(1,n,1)
                self.predictions[i,j,self.lookback_interval*i:] = model.predict(future)[0,:,0]
                self.losses[i,j] = train.history['loss']

                print("Model " + str(counter) + " of " + str(self.N*len(self.histories)) + " trained")
            
            
    
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
        
    # TODO, clean up and add input data.
    def plot_predictions(self,ax = None, title = "Model Predictions: ",xlabel= "Timestep",ylabel="Standardized Price",in_figsize = (5,5),xlim = (None,None)):
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=in_figsize)

        for i in range(len(self.histories)):
          for j in range(self.N):
            ax.plot(self.predictions[i,j][:-self.lookahead],'-k')
            ax.plot(np.linspace(len(self.history[0,:,0])-1,len(self.history[0,:,0])+self.lookahead,self.lookahead+1),self.predictions[i,j][-self.lookahead-1:])
        ax.plot(self.history[0,:,0],'r')
        ax.set_title(title+ self.stockname)
        ax.set_xlim(xlim)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)