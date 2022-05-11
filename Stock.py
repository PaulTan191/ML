import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import Model

## TODO, implement a general LSTM model , using windows.

class Stock:
    
    """
    A class to represent a Stock timeseries with a randomly initizalied LSTM models outputting multiple predictions. Consider making the stock initialization more generalised, and move some of the attributes to method parameters. 
    
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
    
    plot_moving_average(self,window,in_figsize = (5,5), ax = None):
        Plots the moving average of the time series
    
    """
    def __init__(self, history, N, stockname, lookahead = 50, epochs = 1000,theta = [64], lookbacks = 5, lookback_interval = 10,batch_size = 50):


      # Require lookbacks < len(history)
        self.batch_size = batch_size
        self.predictions = np.zeros((lookbacks,N,len(history)+lookahead))
        self.losses = np.zeros((lookbacks,N,epochs))
        self.theta = theta

        
        self.input = np.linspace(0,len(history),len(history)).reshape(1,len(history),1)
        # Standardizing the price, for training purposes.
        self.history = ((history - history.mean())/history.std()).reshape(1,len(history),1)
        self.epochs = epochs
        
        self.lookahead = lookahead
        self.N = N
        self.stockname = stockname
        self.lookback_interval = lookback_interval
        self.histories = [self.history[:,self.lookback_interval * i:,:] for i in range(lookbacks)]

        
    def run(self):
        counter = 0
        for i in range(len(self.histories)):
            for j in range(self.N):
                counter += 1
                model = Model.Model(theta = self.theta)
                
                model.compile(loss =tf.keras.losses.MeanSquaredError(),optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001,epsilon = 0.001,amsgrad = True))

                
                train = model.fit(self.input[:,self.lookback_interval*i:,:]-i*self.lookback_interval,self.histories[i],epochs = self.epochs, batch_size = self.batch_size,verbose = 0)
                
                n =  len(self.histories[i][0,:,0])+self.lookahead
                
                future = np.linspace(0,n,n).reshape(1,n,1)
                self.predictions[i,j,self.lookback_interval*i:] = model.predict(future)[0,:,0]
                self.losses[i,j] = train.history['loss']

                print("Model " + str(counter) + " of " + str(self.N*len(self.histories)) + " trained")
            
    def get_moving_average(self,window):
        arr = np.ones(2*window + 1)/(2*window+1)
        return np.convolve(arr,self.history[0,:,0])
    # If start = None, then use N = lookbacks training subsets of the training data starting from 0, j, 2j, 3j, ... , nj and ending val set timesteps from the end of the time series dataset, With j = lookback_interval
    def train_test(self,val_start = 20, start = None):
      if start != None:
        X_train = self.input[:,start:-val_start,:]
        X_test = self.input[:,-val_start:,:]
        Y_train = self.history[:,start:-val_start,:]
        Y_test = self.history[:,-val_start:,:]
        model = Model.Model(theta = self.theta)        

        model.compile(loss =tf.keras.losses.MeanSquaredError(),optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001,epsilon = 0.001,amsgrad = True))
        train = model.fit(X_train,Y_train,epochs = self.epochs, batch_size = self.batch_size,verbose = 0,validation_data = (X_test,Y_test))
        N = len(self.history[0,start:,0])
        xs = np.linspace(0,N,N)
      
        Y_pred = model.predict(xs.reshape((1,N,1)))[0,:,0]
        # I should rework class attributes and methods better to avoid having to write the following code
        fig, ax = plt.subplots(1,1, figsize = (5,5))

        ax.plot(train.history['loss'],label = "Training Loss")
        ax.plot(train.history['val_loss'],label = "Validation Loss")

        fig2, ax2 = plt.subplots(1,1, figsize = (15,5))

        ax2.plot(X_train[0,:,0],Y_train[0,:,0],label = "Training Data")
        ax2.plot(X_test[0,:,0],Y_test[0,:,0],label = "Validation Data")
        ax2.plot(xs,Y_pred,label = "Prediction")
        ax2.set_ylabel("Standardised Price")

        ax2.legend()
        ax.legend()











    def plot_losses(self,ax = None, title = "Loss",xlabel= "Epochs",ylabel="MSE",in_figsize = (5,5),log = True):
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=in_figsize)
        counter = 0
        for i in range(len(self.histories)):
          for j in range(self.N):
            counter += 1
            ax.plot(self.losses[i,j],label = "Model " + str(counter))
        if log:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.legend()
        
    def plot_predictions(self,ax = None, title = "Model Predictions: ",xlabel= "Timestep",ylabel="Standardized Price",in_figsize = (5,5),xlim = (None,None)):
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=in_figsize)

        counter = 0
        for i in range(len(self.histories)):
          for j in range(self.N):
            counter += 1
            ax.plot(self.predictions[i,j],label = "Model " + str(counter))#[:-self.lookahead])
            #ax.plot(np.linspace(len(self.history[0,:,0])-1,len(self.history[0,:,0])+self.lookahead,self.lookahead+1),self.predictions[i,j][-self.lookahead-1:])
        ax.plot(self.history[0,:,0],'r')
        ax.set_title(title+ self.stockname)
        ax.set_xlim(xlim)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.legend()


    def plot_moving_average(self,window,in_figsize = (5,5), ax = None):
        if ax is None:
          fig , ax = plt.subplots(1,1,figsize = in_figsize)
        ax.plot(self.get_moving_average(window))
        ax.set_title("Moving Average, window size: " + str(window))
        ax.set_ylabel("Standardiced Price")
        ax.set_xlabel("Timesteps in the past")

