#include </usr/local/include/fftw3.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <iostream>

#define REAL 0
#define IMAG 1


static void transpose(fftw_complex *array, int m, int n)
{
    fftw_complex new_array[m * n];
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int old_idx = i * n + j;
            int new_idx = j * m + i;
            new_array[new_idx][REAL] = array[old_idx][REAL];
            new_array[new_idx][IMAG] = array[old_idx][IMAG];
        }
    }
    for (int i = 0; i < m * n; i++)
    {
        array[i][REAL] = new_array[i][REAL];
        array[i][IMAG] = new_array[i][IMAG];
    }
}

void print_matrix_form(fftw_complex *array, const int m, const int n)
{
    for (int outer=0; outer<m; outer++)
    {
        for (int inner=0; inner<n; inner++)
        {
            printf("%.01f + i * %0.01f ", array[inner+outer*n][REAL],
                array[inner+outer*n][IMAG]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void element_wise_multiply(fftw_complex* in1, fftw_complex* in2,
    fftw_complex* out, const int m, const int n)
{
    int index;
    for (int ii=0; ii<m; ii++)
    {
        for (int jj=0; jj<n; jj++)
        {
            index = jj + ii*m;
            out[index][REAL] = in1[index][REAL] * in2[index][REAL]
                - in1[index][IMAG] * in2[index][IMAG];
            out[index][IMAG] = in1[index][REAL] * in2[index][IMAG]
                + in2[index][REAL] * in1[index][IMAG];
        }
    }   
}

void element_wise_multiply_by_KR_grid(fftw_complex *array,
    double *grid, const int m, const int n)
{
    int index;
    for (int ii=0; ii<m; ii++)
    {
        for (int jj=0; jj<n; jj++)
        {
            index = jj + ii*m;
            array[index][REAL] = array[index][REAL] * grid[ii];
            array[index][IMAG] = array[index][IMAG] * grid[ii];
        }
    } 
}

int main(int argc, char const *argv[])
{
    const int m = 3;
    const int n = 4;
    //int a1[m*m];
    //for(int ii=0; ii<m*m; ii++){a1[ii] = ii;}

    double kgrid[m];
    for (int ii=0; ii<m; ii++){kgrid[ii] = 2.0;}

    fftw_complex a1[m*n], a2[m*n], out[m*n];
    for(int ii=0; ii<m*n; ii++){a1[ii][REAL] = ii; a1[ii][IMAG] = -ii;}
    for(int ii=0; ii<m*n; ii++){a2[ii][REAL] = ii-1; a2[ii][IMAG] = ii+1;}
    for(int ii=0; ii<m*n; ii++){std::cout << a1[ii][REAL] << std::endl;}
    print_matrix_form(a1, m, n);
    transpose(a1, m, n);
    print_matrix_form(a1, n, m);
    transpose(a1, n, m);
    print_matrix_form(a1, m, n);
    std::cout << "---" << std::endl;

    element_wise_multiply(a1, a2, out, m, n);
    print_matrix_form(a1, m, n);
    print_matrix_form(a2, m, n);
    print_matrix_form(out, m, n);

    return 0;
}
