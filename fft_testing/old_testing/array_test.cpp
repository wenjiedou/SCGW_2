#include </usr/local/include/fftw3.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <iostream>

#define NR 100
#define DR 0.12315

#define NT 1000
#define DT 0.1

#define SD 0.1
#define EF 0.25

#define REAL 0
#define IMAG 1

void generate_gaussian_signal(fftw_complex* signal, double* wgrid)
{
    double mu;
    for (int kk=0; kk<NR; kk++)
    {
        mu = -5.0+kk*kk*0.01;
        for (int ww=0; ww<NT; ww++)
        {
            double x = wgrid[ww];
            signal[ww+NT*kk][REAL] = exp(-(x-mu)*(x-mu)/2.0/SD/SD)/SD/sqrt(2.0*M_PI);
            signal[ww+NT*kk][IMAG] = 0.0;
        }
    }
}

void writefile(fftw_complex* result, double* grid, const int kindex,
    std::string fname)
{
    FILE *f1;
    f1 = fopen(fname.c_str(), "w");
    for (int ii=0; ii < NT; ii++)
    {
        fprintf(f1, "%f %f %f\n", grid[ii], result[ii+NT*kindex][REAL],
            result[ii+NT*kindex][IMAG]);
    }
    fclose(f1);
}


void normalize_results_k(fftw_complex* result)
{
    for (int ii=0; ii<NT*NR; ii++)
    {
        result[ii][REAL] = result[ii][REAL] * DT;
        result[ii][IMAG] = result[ii][IMAG] * DT;
    }
}

void normalize_results_r(fftw_complex* result)
{
    for (int ii=0; ii<NT*NR; ii++)
    {
        result[ii][REAL] = result[ii][REAL] / NT / DT;
        result[ii][IMAG] = result[ii][IMAG] / NT / DT;
    }
}


int main(int argc, char const *argv[])
{

    const int kstar = 50;

    fftw_complex signal[NT*NR];

    // Define the grids
    double tgrid[NT], wgrid[NT];
    for (int ii=0; ii<int(NT/2); ii++)
    {
        tgrid[ii] = ii * DT;
        wgrid[ii] = double(ii * 2 * M_PI / NT / DT);
    }


    int ccR = 0;
    int cc = int(NT/2);
    for (int ii=int(NT/2); ii>0; ii--)
    {
        tgrid[cc] = (-int(NT/2) + ccR) * DT;
        wgrid[cc] = -double(ii * 2 * M_PI / NT / DT);
        cc++;
        ccR++;
    }

    generate_gaussian_signal(signal, wgrid);
    writefile(signal, wgrid, kstar, "test.txt");

    fftw_complex *in_forward = (fftw_complex*) fftw_malloc(NT*sizeof(fftw_complex));    
    fftw_plan forward_plan = fftw_plan_dft_1d(NT, in_forward, in_forward,
        FFTW_FORWARD, FFTW_MEASURE);

    fftw_complex *in_backward = (fftw_complex*) fftw_malloc(NT*sizeof(fftw_complex));    
    fftw_plan backward_plan = fftw_plan_dft_1d(NT, in_backward, in_backward,
        FFTW_BACKWARD, FFTW_MEASURE);

    for(int iGrid=0;iGrid<NR;iGrid++)
    {
        memcpy(in_backward,&signal[iGrid*NT],NT*sizeof(fftw_complex));
        fftw_execute(backward_plan);
        memcpy(&signal[iGrid*NT],in_backward,NT*sizeof(fftw_complex));
    }

    normalize_results_r(signal);
    writefile(signal, tgrid, kstar, "test_fft.txt");

    for(int iGrid=0;iGrid<NR;iGrid++)
    {
        memcpy(in_forward,&signal[iGrid*NT],NT*sizeof(fftw_complex));
        fftw_execute(forward_plan);
        memcpy(&signal[iGrid*NT],in_forward,NT*sizeof(fftw_complex));
    }

    normalize_results_k(signal);
    writefile(signal, wgrid, kstar, "test_fft_recovered.txt");

    return 0;
}
