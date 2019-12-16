// Based off of https://github.com/undees/fftw-example/blob/master/fftw_example.c

#include </usr/local/include/fftw3.h>

#define NUM_POINTS 10940
#define DR 0.12315


/* Never mind this bit */

#include <stdio.h>
#include <math.h>
#include <complex.h>

#define REAL 0
#define IMAG 1

void acquire_from_somewhere(fftw_complex* signal, double* rgrid) {
    int i;
    const double mu = 1.0;
    const double sd = 1.0;
    for (i = 0; i < NUM_POINTS*2; ++i) {
        double x = rgrid[i];
        signal[i][REAL] = exp(-(x-mu)*(x-mu)/2.0/sd/sd)/sd/sqrt(2.0*M_PI);
        signal[i][IMAG] = 0.0;
    }
}

void writefile(fftw_complex* result, double* grid, std::string fname){
    FILE *f1;
    f1 = fopen(fname.c_str(), "w");
    for (int ii=0; ii < NUM_POINTS*2; ii++)
    {
        fprintf(f1, "%f %f %f\n", grid[ii], result[ii][REAL], result[ii][IMAG]);
    }
    fclose(f1);
}

void normalize_results_k(fftw_complex* result)
{
    for (int ii=0; ii<NUM_POINTS*2; ii++)
    {
        result[ii][REAL] = result[ii][REAL] * DR;
        result[ii][IMAG] = result[ii][IMAG] * DR;
    }
}

void normalize_results_r(fftw_complex* result)
{
    for (int ii=0; ii<NUM_POINTS*2; ii++)
    {
        result[ii][REAL] = result[ii][REAL] / NUM_POINTS / 2 / DR;
        result[ii][IMAG] = result[ii][IMAG] / NUM_POINTS / 2 / DR;
    }
}

int main() {
    fftw_complex signal[NUM_POINTS*2];
    fftw_complex result[NUM_POINTS*2];
    fftw_complex signal_recovered[NUM_POINTS*2];

    // Define the grids
    double rgrid[int(NUM_POINTS*2)], kgrid[int(NUM_POINTS*2)];
    for (int ii=0; ii<NUM_POINTS; ii++)
    {
        rgrid[ii] = ii * DR;
        kgrid[ii] = double(ii * 2 * M_PI / NUM_POINTS / 2 / DR);
    }

    int ccR = 0;
    int cc = NUM_POINTS;
    for (int ii=NUM_POINTS; ii>0; ii--)
    {
        rgrid[cc] = (-NUM_POINTS + ccR) * DR;
        kgrid[cc] = -double(ii * 2 * M_PI / NUM_POINTS / 2 / DR);
        cc++;
        ccR++;
    }

    acquire_from_somewhere(signal, rgrid);
    writefile(signal, rgrid, "gaussian_r_initial.txt");

    fftw_plan plan = fftw_plan_dft_1d(NUM_POINTS*2,
                                      signal,
                                      result,
                                      FFTW_FORWARD,
                                      FFTW_ESTIMATE);

    fftw_execute(plan);
    normalize_results_k(result);
    writefile(result, kgrid, "gaussian_k.txt");
    fftw_destroy_plan(plan);

    // And inverse FFT
    fftw_plan inv_plan = fftw_plan_dft_1d(NUM_POINTS*2,
                                          result,
                                          signal_recovered,
                                          FFTW_BACKWARD,
                                          FFTW_ESTIMATE);
    fftw_execute(inv_plan);
    normalize_results_r(signal_recovered);
    writefile(signal_recovered, rgrid, "gaussian_r_recovered.txt");
    fftw_destroy_plan(inv_plan);

    return 0;
}
