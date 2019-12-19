#include </usr/local/include/fftw3.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <iostream>

#define REAL 0
#define IMAG 1


static void symmetrize_over_0th_axis(fftw_complex *array, int m, int n)
{

    int swap_index = 1;
    const int upper = int(m/2+1);
    for (int ii=upper; ii<m; ii++)
    {
        for (int jj=0; jj<n; jj++)
        {
            array[jj + ii*n][REAL] = -array[jj + (upper-swap_index)*n][REAL];
            array[jj + ii*n][IMAG] = -array[jj + (upper-swap_index)*n][IMAG];
        }
        swap_index += 1;
    }
}

void print_matrix_form(fftw_complex *array, const int m, const int n)
{
    for (int outer=0; outer<m; outer++)
    {
        for (int inner=0; inner<n; inner++)
        {
            printf("%.01f ", array[inner+outer*n][REAL]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}



int main(int argc, char const *argv[])
{
    const int m = 9;
    const int n = 4;
    //int a1[m*m];
    //for(int ii=0; ii<m*m; ii++){a1[ii] = ii;}
    fftw_complex a1[m*n];
    for(int ii=0; ii<m*n; ii++){a1[ii][REAL] = ii; a1[ii][IMAG] = 0.0;}
    print_matrix_form(a1, m, n);
    symmetrize_over_0th_axis(a1, m, n);
    print_matrix_form(a1, m, n);



    return 0;
}
