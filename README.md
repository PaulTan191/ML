# ML
Simple collection of ML projects. 

1. Final Project is a machine learning project completed for an National University of Singapore ML course: "Machine Learning for Physicists". It is a simple project, where I investigate the structure of the periodic table.
2. Stockus.ipynb is a work in progress jupyter notebook, where I am working on predicting stock prices.
3. Stock.py contains the classes defined for use in the notebook
4. Model.py contains a subclass for a simple (perhaps naive implementation) of an LSTM model.

## Stock Prediction Project: Todo and Ideas

Stock.py should contain methods and attributes which are suitable to be instantiated in a broader survey of multiple stocks.

1. Refine implementation of LSTM model by first learning properly the deatils of RNN and afterwards LSTM. In particular the use of windows in the input data should be understood and if applicable implemented. This should tie into batch sizes. 
2. Implement a test-validation split. I have rushed past the basics. Then, write methods which give summary statistics and numeric results for which a model prediction can be evaluated. For example: training loss, validation loss, variance of multiple predictions (to test reliability). 
3. Evaluate the effect of using log returns against different metrics.
4. As EDA, plot the running average/convolution of stock prices/log returns; as a form of denoising, evaluate the effectiveness of modelling a running average instead of daily metrics. By denoising, model complexity should decrease, too. Consider that a non RNN model may be sufficient for such a denoised timeseries.
5. Over time, stocks can enter different phases, e.g volatility may change, the average may drastically change, it may be worth developing metrics which measure these changes and change the training set as a result.
