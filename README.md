# Time Series Analysis in Python


**TSAP** is a python package that provides tools for time series analysis in
financial data.

Given input of a stock price series, the system will fit time series models,
estimate the parameters and do statistical inference. With the identified model,
the system predict the future price and assess the prediction accuracy.  We can
further consider trading strategy and option pricing. Moreover, given the input
of multiple stock prices, the system can implement clustering and build a
reduced order model for price prediction.


## Installation

1. Download TSAP package from GitHub: 
  `git clone https://github.com/APC524/tsap.git`
2. Add the folder [tsap](https://github.com/APC524/tsap/tree/master/tsap) into
   your Python search path.


## Functionality

TSAP package provides six Python classes.
1. **AR**: the autoregressive model to fit the imput stock price series,
   computing the log-likelihood and the gradient.
2. **MR**: the moving average model.
3. **Solver**: estimate the model parameters given the model class and the
   optimization method.
4. OptionPricing: calculate the option price given the underlying stock.
5. Cluster: impelement the clustering of multiple stock price series.
6. Reduction: build a reduced order model for price prediction. 

Following is the high-level program structure figure.
![Program structure](https://github.com/APC524/tsap/blob/master/doc/report/Figure/structure.png)


## Documents and demos

* The [Project Report](https://github.com/APC524/tsap/blob/master/doc/report/report.pdf)
  explains the detail of the whole project.
* The [User Manual](https://github.com/APC524/tsap/blob/master/doc/manual/manual.pdf)
  gives a brief introduction of the functionality of the package.
* The user can also generate Doxygen HTML and LaTeX manuals with
  [Doxyfile](https://github.com/APC524/tsap/blob/master/doc/manual/Doxyfile),
  using the command `doxygen Doxyfile`.
* In [demo](https://github.com/APC524/tsap/tree/master/demo) folder there are
  several examples showing how to use the package.


## Contributors

This is the course project of *APC524/MAE560 Software Engineering for Scientific
Computing* (Fall 2016) in Princeton University. The project members are Wenyan
Gong, Zongxi Li, Cong Ma, Qingcan Wang, Zhuoran Yang and Hao Zhang. We would
appreciate Professor Stone and Assistant Instructor Jeffry and Bernat for their
guidance and help.
