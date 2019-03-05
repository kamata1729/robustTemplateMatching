#include "cython_calc_NCC.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void calc_NCC(double *m_flat, int m_size[3], double *f_flat, int f_size[3], double *NCC_flat)
{
    // calc |F|
    printf("calc F\n");
    double f_norm = 0.0;
    for(int i=0; i<f_size[0]*f_size[1]*f_size[2]; i++)
    {
        f_norm += pow(f_flat[i], 2);
    }
    f_norm = sqrt(f_norm);
    printf("f_norm\n");
    printf("%f\n", f_norm);
    for (int j = 0; j < m_size[1] - f_size[1]; j++)
    {
        for (int k = 0; k < m_size[2] - f_size[2]; k++)
        {
            for (int i = 0; i < m_size[0]; i++)
            {
                double m_norm = 0.0;
                for (int u = 0; u < f_size[1]; u++)
                {
                    for (int v = 0; v < f_size[2]; v++)
                    {
                        NCC_flat[j * (m_size[2] - f_size[2]) + k] += m_flat[i * m_size[1] * m_size[2] + (j + u) * m_size[2] + k + v] * f_flat[i * f_size[1] * f_size[2] + u * f_size[2] + v];
                        m_norm += pow(m_flat[i * m_size[1] * m_size[2] + (j + u) * m_size[2] + k + v], 2);
                    }
                }
                NCC_flat[j * (m_size[2] - f_size[2]) + k] /= (f_norm * sqrt(m_norm));
            }
        }
    }
    printf("fin F\n");

}