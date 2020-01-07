#include </usr/local/include/fftw3.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <iostream>

// Note NR and NT should be ODD!!!

#define NR 11
#define DR 1.0

#define NT 1001
#define DT 0.1

#define SD 0.5
#define EF 0.25
#define KF 0.5

#define REAL 0
#define IMAG 1

void generate_gaussian_signal(fftw_complex* signal, double* wgrid,
    double* kgrid)
{
    double mu;
    for (int kk=0; kk<NR; kk++)
    {
        mu = kgrid[kk]*kgrid[kk] - EF;
        for (int ww=0; ww<NT; ww++)
        {
            double x = wgrid[ww];
            signal[ww+NT*kk][REAL] = exp(-(x-mu)*(x-mu)/2.0/SD/SD)/SD/sqrt(2.0*M_PI);
            signal[ww+NT*kk][IMAG] = 0.0;
        }
        //std::cout << kk << " " << kgrid[kk] << std::endl;
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

int get_kf_index(double* kgrid)
{
    double min_dist = 1e16;
    double dist;
    int istar = -1;
    for (int ii=0; ii<NR; ii++)
    {
        dist = fabs(kgrid[ii] - KF);
        if (dist < min_dist)
        {
            istar = ii;
            min_dist = dist;
        }
    }
    return istar;
}

static void init_grids(const int numR, const double dR, const int numT,
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

int main(int argc, char const *argv[])
{

    fftw_complex signal[NT*NR];

    // Define the grids
    double tgrid[NT], wgrid[NT], rgrid[NR], kgrid[NR];
    init_grids(NR, DR, NT, DT, tgrid, wgrid, rgrid, kgrid);

    const int kstar = get_kf_index(kgrid);
    printf("kf index is %i (k=%.05f)\n", kstar, kgrid[kstar]);

    generate_gaussian_signal(signal, wgrid, kgrid);
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
