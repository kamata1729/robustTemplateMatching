cimport numpy as np

cdef extern from "cython_calc_NCC.h":
    void calc_NCC(float *m_flat, int m_size[3], float *f_flat, int f_size[3], float *NCC_flat)

def c_calc_NCC(np.ndarray[float, ndim=1] m_flat, np.ndarray[int, ndim=1] m_size, np.ndarray[float, ndim=1] f_flat, np.ndarray[int, ndim=1] f_size, np.ndarray[float, ndim=1] NCC_flat):
    calc_NCC(&m_flat[0], &m_size[0], &f_flat[0], &f_size[0], &NCC_flat[0])