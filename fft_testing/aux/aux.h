#include </usr/local/include/fftw3.h>
#include <iostream>

#ifndef AUX_H
#define AUX_H

void symmetrize_over_0th_axis(fftw_complex *array, int m, int n);
void print_matrix_form(fftw_complex *array, const int m, const int n);
void transpose(fftw_complex *array, int m, int n);
void element_wise_multiply(const fftw_complex* in1, const fftw_complex* in2,
    fftw_complex* out, const int m, const int n);
void element_wise_multiply_by_KR_grid(fftw_complex* array,
    const double* grid, const int m, const int n);
void element_wise_divide_by_KR_grid(fftw_complex* array,
    const double* grid, const int m, const int n);
void init_grids(const int numR, const double dR, const int numT,
    const double dT, double* tgrid, double* wgrid, double* rgrid,
    double* kgrid);
void write_row(fftw_complex* result, const double* grid, const int kindex,
    const int n_row, std::string fname);
int get_kf_index(double* kgrid, const int nR);

#endif
