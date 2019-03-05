import numpy as np
import cython_calc_NCC

if __name__ == '__main__':
    a = np.random.randint(0, 10, (2,6,7))
    b = np.random.randint(0, 10, (2, 3, 4))
    print("a")
    print(a)
    print("b")
    print(b)
    print("NORM F")
    print(np.linalg.norm(b))
    c =  np.zeros(a.shape[0] * (a.shape[1] - b.shape[1]) * (a.shape[2] - b.shape[2])).astype(np.float32)
    cython_calc_NCC.c_calc_NCC(a.flatten().astype(np.float32), np.array(a.shape).astype(
        np.int32), b.flatten().astype(np.float32), np.array(b.shape).astype(np.int32), c)
    print("c")
    print(c)
    print(type(c))