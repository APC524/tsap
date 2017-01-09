import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern void c_ar1_gen(double * array, const double rho, const double sigma, const int time_, const int num, const  int burnin)

@cython.boundscheck(False)
@cython.wraparound(False)
def ar1_gen(double rho, double sigma, int time, int num, int burnin):
    """
    generate AR(1) data, rho is the coefficient

    """
    cdef np.ndarray[double, ndim=2, mode="c"] data = np.zeros( (num, time + burnin))

    c_ar1_gen (&data[0,0], rho, sigma, time, num, burnin )

    out = data[:, burnin: ]

    return out


# declare the interface to the C code
cdef extern void c_ma1_gen(double * array, const double rho, const double constant, const int time, const int num, const int burnin )

@cython.boundscheck(False)
@cython.wraparound(False)
def ma1_gen( double rho, double constant,  int time,  int num,  int burnin ):
    """
    generate MA(1) data,   X_t = constant + e_t + rho * e_{t-1}
    """

    cdef np.ndarray[double, ndim=2, mode="c"] data = np.zeros( (num, time + burnin))

    c_ma1_gen(&data[0,0], rho, constant, time, num, burnin)
    out = data[:, burnin: ]
    return out


cdef extern void c_arch1_gen(double *array, double a0, double a1, int time, int num, int burnin)

@cython.boundscheck(False)
@cython.wraparound(False)
def arch1_gen(double a0, double a1, int time_, int num, int burnin):
    """
    generate ARCH(1) data
    """

    cdef np.ndarray[double, ndim=2, mode="c"] data = np.zeros( (num, time + burnin))

    c_arch1_gen(&data[0,0], a0, a1, time, num, burnin)
    out = data[:, burnin:]
    return out


cdef extern void c_garch11_gen(double * array, double a, double b, double c, const int time_, const int num, const int burnin )

@cython.boundscheck(False)
@cython.wraparound(False)
def garch11_gen(double a, double b, const int time_, const int num, const int burnin ):
    """ generate data from GARCH(1,1) model
    """

    cdef np.ndarray[double, ndim=2, mode="c"] data = np.zeros( (num, time + burnin))

    c_garch11_gen(&data[0,0], a, b, c, time, num, burnin)
    out = data[:, burnin:]
    return out
