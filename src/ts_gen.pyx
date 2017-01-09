import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern void c_ar1_gen(double* array, double rho, double sigma, int time, int num, int burnin )

@cython.boundscheck(False)
@cython.wraparound(False)
def ar1_gen(double rho, double sigma, int time, int num, int burnin):
    """
    generate AR(1) data, rho is the coefficient

    """
    cdef np.ndarray[double, ndim=2, mode="c"] data = np.zeros( (num, time + burnin))

    c_ar1_gen (&data[0,0], rho, sigma, time, num, burnin )

    #data = np.ascontiguousarray(data, dtype = np.float64);
    #print(data.flags)
    out = data[:, burnin: ]

    return out
