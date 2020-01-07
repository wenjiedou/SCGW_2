#include </usr/local/include/fftw3.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <iostream>

#define REAL 0
#define IMAG 1


void symmetrize_over_0th_axis(fftw_complex *array, int m, int n)
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

void transpose(fftw_complex *array, int m, int n)
{
    fftw_complex *new_array;
    new_array = (fftw_complex*) fftw_malloc(m*n*sizeof(fftw_complex));
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
    fftw_free(new_array);
}

void element_wise_multiply(const fftw_complex* in1, const fftw_complex* in2,
    fftw_complex* out, const int m, const int n)
{
    int index;
    for (int ii=0; ii<m; ii++) // number of rows
    {
        for (int jj=0; jj<n; jj++) // number of columns
        {
            index = jj + ii*n;
            out[index][REAL] = in1[index][REAL] * in2[index][REAL]
                - in1[index][IMAG] * in2[index][IMAG];
            out[index][IMAG] = in1[index][REAL] * in2[index][IMAG]
                + in2[index][REAL] * in1[index][IMAG];
        }
    }   
}

void element_wise_multiply_by_constant(fftw_complex* array,
    const double c, const int m, const int n)
{
    int index;
    for (int ii=0; ii<m; ii++) // number of rows
    {
        for (int jj=0; jj<n; jj++) // number of columns
        {
            index = jj + ii*n;
            array[index][REAL] = array[index][REAL] * c;
            array[index][IMAG] = array[index][IMAG] * c;
        }
    }   
}

void element_wise_multiply_by_KR_grid(fftw_complex* array,
    const double* grid, const int m, const int n)
{
    int index;
    for (int ii=0; ii<m; ii++)
    {
        for (int jj=0; jj<n; jj++)
        {
            index = jj + ii*n;
            array[index][REAL] = array[index][REAL] * grid[ii];
            array[index][IMAG] = array[index][IMAG] * grid[ii];
        }
    } 
}

void element_wise_divide_by_KR_grid(fftw_complex* array,
    const double* grid, const int m, const int n)
{
    int index;
    for (int ii=0; ii<m; ii++)
    {
        for (int jj=0; jj<n; jj++)
        {
            index = jj + ii*n;
            array[index][REAL] = array[index][REAL] / grid[ii];
            array[index][IMAG] = array[index][IMAG] / grid[ii];
        }
    } 
}

void init_grids(const int numR, const double dR, const int numT,
    const double dT, double* tgrid, double* wgrid, double* rgrid,
    double* kgrid)
{
    for (int ii=0; ii<numT; ii++)
    {
        if (ii < numT/2+1)
        {
            tgrid[ii] = ii * dT;
            wgrid[ii] = ii * 2.0 * M_PI / numT / dT;    
        }
        
        else
        {
            tgrid[ii] = (ii - numT) * dT;
            wgrid[ii] = (ii - numT) * 2.0 * M_PI / numT / dT;
        }
    }

    for (int ii=0; ii<numR; ii++)
    {
        if (ii < numR/2+1)
        {
            rgrid[ii] = ii * dR;
            kgrid[ii] = ii * 2.0 * M_PI / numR / dR;    
        }
        
        else
        {
            rgrid[ii] = (ii - numR) * dR;
            kgrid[ii] = (ii - numR) * 2.0 * M_PI / numR / dR;
        }
    }
}

void write_row(fftw_complex* result, const double* grid, const int kindex,
    const int n_row, std::string fname)
{
    FILE *f1;
    f1 = fopen(fname.c_str(), "w");
    for (int ii=0; ii < n_row; ii++)
    {
        fprintf(f1, "%f %f %f\n", grid[ii], result[ii+n_row*kindex][REAL],
            result[ii+n_row*kindex][IMAG]);
    }
    fclose(f1);
}

int get_kf_index(double* kgrid, const int nR)
{
    double min_dist = 1e16;
    double dist;
    int istar = -1;
    for (int ii=0; ii<nR; ii++)
    {
        dist = fabs(kgrid[ii] - 0.5);
        if (dist < min_dist)
        {
            istar = ii;
            min_dist = dist;
        }
    }
    return istar;
}
