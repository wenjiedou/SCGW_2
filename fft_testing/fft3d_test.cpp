#include <complex.h>
#include </usr/local/include/fftw3.h>
#include <stdio.h>
#include <math.h>
#include <iostream>

#include "aux/aux.h"

#define NR 101
#define DR 1.0

#define NT 5001
#define DT 0.1

#define SD 0.5
#define EF 0.25
#define KF 0.5

#define ALPHA 0.521062
#define RS 4.0

#define REAL 0
#define IMAG 1

void get_oneshot_G_kw_gre_G_les(fftw_complex *G_gre0,
    fftw_complex *G_les0, const double* wgrid, const double* kgrid)
{
    double mu, x;
    fftw_complex gsignal[NT*NR];
    for (int kk=0; kk<NR; kk++)
    {
        mu = kgrid[kk]*kgrid[kk] - EF;
        for (int ww=0; ww<NT; ww++)
        {
            x = wgrid[ww];

            // Note the -2 "normalization" factor for A
            gsignal[ww+NT*kk][REAL] = 
                -2.0 * exp(-(x-mu)*(x-mu)/2.0/SD/SD)/SD/sqrt(2.0*M_PI);
            gsignal[ww+NT*kk][IMAG] = 0.0;
        }
    }

    double wval;
    for (int kk=0; kk<NR; kk++)
    {
        for (int ww=0; ww<NT; ww++)
        {
            wval = wgrid[ww];
            if (wval > 0.0)
            {
                G_les0[ww+NT*kk][REAL] = 0.0;
                G_les0[ww+NT*kk][IMAG] = gsignal[ww+NT*kk][REAL];
                G_gre0[ww+NT*kk][REAL] = 0.0;
                G_gre0[ww+NT*kk][IMAG] = 0.0;
            }
            else
            {
                G_les0[ww+NT*kk][REAL] = 0.0;
                G_les0[ww+NT*kk][IMAG] = 0.0;
                G_gre0[ww+NT*kk][REAL] = 0.0;
                G_gre0[ww+NT*kk][IMAG] = -gsignal[ww+NT*kk][REAL];
            }
        }
    }
}

void get_P_rt(const fftw_complex* G_les, const fftw_complex* G_gre,
    fftw_complex* P_rt, const double* tgrid)
{
    int ii;
    for (int rr=0; rr<NR; rr++)
    {
        for (int tt=0; tt<NT; tt++)
        {   
            ii = tt + NT*rr;

            if (tgrid[tt] > 0.0)
            {
                P_rt[ii][REAL] =
                    4.0 * (G_gre[ii][IMAG]*G_les[ii][REAL]
                           - G_gre[ii][REAL]*G_les[ii][IMAG]);
            }
            else
            {
                P_rt[ii][REAL] = 0.0;
            }
            P_rt[ii][IMAG] = 0.0;
            
        }
    }
}

static void normalize_results_r_to_k(fftw_complex* result)
{
    for (int ii=0; ii<NT*NR; ii++)
    {
        result[ii][REAL] = result[ii][REAL] * DR;
        result[ii][IMAG] = result[ii][IMAG] * DR;
    }
}

static void normalize_results_k_to_r(fftw_complex* result)
{
    for (int ii=0; ii<NT*NR; ii++)
    {
        result[ii][REAL] = result[ii][REAL] / NR / DR;
        result[ii][IMAG] = result[ii][IMAG] / NR / DR;
    }
}

static void normalize_results_t_to_w(fftw_complex* result)
{
    for (int ii=0; ii<NT*NR; ii++)
    {
        result[ii][REAL] = result[ii][REAL] * DT;
        result[ii][IMAG] = result[ii][IMAG] * DT;
    }
}

static void normalize_results_w_to_t(fftw_complex* result)
{
    for (int ii=0; ii<NT*NR; ii++)
    {
        result[ii][REAL] = result[ii][REAL] / NT / DT;
        result[ii][IMAG] = result[ii][IMAG] / NT / DT;
    }
}

void inplace_ifft_kw_to_rt(fftw_complex* signal, fftw_plan plan_R,
    fftw_plan plan_T, fftw_complex* memLoc_R, fftw_complex* memLoc_T, 
    const double* kgrid, const double* rgrid)
{
    // First, multiply every row by the correct k-value
    element_wise_multiply_by_KR_grid(signal, kgrid, NR, NT);

    // Symmetrize over the k-axis
    symmetrize_over_0th_axis(signal, NR, NT);    

    // Execute iFFT with normalization: k, w -> k, t
    for(int iGrid=0;iGrid<NR;iGrid++)
    {
        memcpy(memLoc_T, &signal[iGrid*NT], NT*sizeof(fftw_complex));
        fftw_execute(plan_T);
        memcpy(&signal[iGrid*NT], memLoc_T, NT*sizeof(fftw_complex));
    }
    normalize_results_w_to_t(signal);

    // Transpose the matrix (note that here the second argument is the
    // number of rows and the third is the number of columns, this will
    // be switched later)
    transpose(signal, NR, NT);

    // Execute iFFT with normalization: k, t -> r, t
    for (int iGrid=0; iGrid<NT; iGrid++)
    {
        memcpy(memLoc_R, &signal[iGrid*NR], NR*sizeof(fftw_complex));
        fftw_execute(plan_R);
        memcpy(&signal[iGrid*NR], memLoc_R, NR*sizeof(fftw_complex));
    }
    normalize_results_k_to_r(signal);
    transpose(signal, NT, NR);
    element_wise_divide_by_KR_grid(signal, rgrid, NR, NT);
}

void inplace_fft_rt_to_kw(fftw_complex* signal, fftw_plan plan_R,
    fftw_plan plan_T, fftw_complex* memLoc_R, fftw_complex* memLoc_T, 
    const double* kgrid, const double* rgrid)
{
    // First, multiply every row by the correct r-value
    element_wise_multiply_by_KR_grid(signal, rgrid, NR, NT);

    // Symmetrize over the k-axis
    symmetrize_over_0th_axis(signal, NR, NT);    

    // Execute FFT with normalization: r, t -> r, w
    for(int iGrid=0;iGrid<NR;iGrid++)
    {
        memcpy(memLoc_T, &signal[iGrid*NT], NT*sizeof(fftw_complex));
        fftw_execute(plan_T);
        memcpy(&signal[iGrid*NT], memLoc_T, NT*sizeof(fftw_complex));
    }
    normalize_results_t_to_w(signal);

    // Transpose the matrix (note that here the second argument is the
    // number of rows and the third is the number of columns, this will
    // be switched later)
    transpose(signal, NR, NT);

    // Execute iFFT with normalization: k, t -> r, t
    for (int iGrid=0; iGrid<NT; iGrid++)
    {
        memcpy(memLoc_R, &signal[iGrid*NR], NR*sizeof(fftw_complex));
        fftw_execute(plan_R);
        memcpy(&signal[iGrid*NR], memLoc_R, NR*sizeof(fftw_complex));
    }
    normalize_results_r_to_k(signal);
    transpose(signal, NT, NR);
    element_wise_divide_by_KR_grid(signal, kgrid, NR, NT);
}


int main(int argc, char const *argv[])
{

    // Define the grids
    double *tgrid, *wgrid, *rgrid, *kgrid;
    tgrid = new double[NT];
    wgrid = new double[NT];
    rgrid = new double[NR];
    kgrid = new double[NR];

    init_grids(NR, DR, NT, DT, tgrid, wgrid, rgrid, kgrid);
    kgrid[0] = 1e-6;
    rgrid[0] = 1e-6;

    const int kf_index = get_kf_index(kgrid, NR);
    printf("kf index is %i (k=%.05f)\n", kf_index, kgrid[kf_index]);

    // Allocate
    fftw_complex *G_gre0, *G_les0, *P;
    G_gre0 = (fftw_complex*) fftw_malloc(NR*NT*sizeof(fftw_complex));
    G_les0 = (fftw_complex*) fftw_malloc(NR*NT*sizeof(fftw_complex));
    P = (fftw_complex*) fftw_malloc(NR*NT*sizeof(fftw_complex));

    // Construct simple forward and backwards 1d plans, allocate the
    // appropriate contiguous memory
    fftw_complex *in_forward_R =
        (fftw_complex*) fftw_malloc(NR*sizeof(fftw_complex));    
    fftw_plan forward_plan_R = fftw_plan_dft_1d(NR, in_forward_R, in_forward_R,
        FFTW_FORWARD, FFTW_MEASURE);
    fftw_complex *in_backward_R = 
        (fftw_complex*) fftw_malloc(NR*sizeof(fftw_complex));    
    fftw_plan backward_plan_R = fftw_plan_dft_1d(NR, in_backward_R,
        in_backward_R, FFTW_BACKWARD, FFTW_MEASURE);

    fftw_complex *in_forward_T =
        (fftw_complex*) fftw_malloc(NT*sizeof(fftw_complex));    
    fftw_plan forward_plan_T = fftw_plan_dft_1d(NT, in_forward_T, in_forward_T,
        FFTW_FORWARD, FFTW_MEASURE);
    fftw_complex *in_backward_T = 
    (fftw_complex*) fftw_malloc(NT*sizeof(fftw_complex));    
    fftw_plan backward_plan_T = fftw_plan_dft_1d(NT, in_backward_T,
        in_backward_T, FFTW_BACKWARD, FFTW_MEASURE);

    // Start the GW calculation
    // Generate the intial data from Gaussians on a grid
    get_oneshot_G_kw_gre_G_les(G_gre0, G_les0, wgrid, kgrid);
    
    // FT G</>(k, w) from KW -> RT space
    inplace_ifft_kw_to_rt(G_gre0, backward_plan_R, backward_plan_T,
        in_backward_R, in_backward_T, kgrid, rgrid);
    inplace_ifft_kw_to_rt(G_les0, backward_plan_R, backward_plan_T,
       in_backward_R, in_backward_T, kgrid, rgrid);
    
    //Compute P(r, t) = GG
    get_P_rt(G_les0, G_gre0, P, tgrid);
    inplace_fft_rt_to_kw(P, forward_plan_R, forward_plan_T, in_forward_R,
        in_forward_T, kgrid, rgrid);
    write_row(P, wgrid, kf_index, NT, "P0_kw.txt");


    // Cleanup
    fftw_destroy_plan(forward_plan_R);
    fftw_destroy_plan(forward_plan_T);
    fftw_destroy_plan(backward_plan_R);
    fftw_destroy_plan(backward_plan_T);
    fftw_free(G_les0);
    fftw_free(G_gre0);
    fftw_free(P);
    delete [] tgrid;
    delete [] wgrid;
    delete [] rgrid;
    delete [] kgrid;
    
    return 0;
}
