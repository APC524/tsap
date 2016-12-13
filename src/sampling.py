# sample simulated data from AR, MA, VAR models

def ar_gen( lag, num_data, time):
""" generate simulated data from AR model
    lag is an array that contains the coefficients of the autoregressive model
    for AR(p) model, we consider
    X_t = c0 + c_p X_{t-p} + c_{p-1} X_{t-p+1} + ... + c_1 X_{t-1} + epsilon_t,
    where epsilon_t is the random noise, which is assumed to be a Gaussian white noise
    The variance of the white noise is given by sigma

    Output:
        Data: a num_data by time numpy 2d array
        lag: an array that stores the true coefficients
        sigma: the variance of white noise

        The reason we store the model parameters is to evaluate the performance of estimation

"""




