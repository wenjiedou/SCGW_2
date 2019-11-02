#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <complex.h>
#include <fftw3.h>
#include <mkl.h>
#include <string.h>
#include <omp.h>
#include "gw.h"

/* One iteration GW calculation */
void gwOneIteration(GWPARAM *gwParam){
  int iGrid,jGrid,kGrid;
  int index;
  int numGridK = gwParam->numGridK;
  int numGridR = gwParam->numGridR;
  int numGridT = gwParam->numGridT;
  int numGridW = gwParam->numGridW;
  int numGridProcK = gwParam->numGridProcK;
  int numGridProcR = gwParam->numGridProcR;
  int numGridProcT = gwParam->numGridProcT;
  int numGridProcW = gwParam->numGridProcW;
  int myid = gwParam->myid;

  double mu = gwParam->mu;
  double w,k;
  double dw = gwParam->dw;
  double alpha_rs = gwParam->alpha_rs;
  double epsilonF = gwParam->epsilonF;

  double *wGrid = gwParam->wGrid;
  double *tGrid = gwParam->tGrid;
  double *thetaW_mu = gwParam->thetaW_mu;
  double *thetaW = gwParam->thetaW;
  double *thetaT = gwParam->thetaT;
  double *vkGridProc = gwParam->vkGridProc;
  double *epsilonKGridProc = gwParam->epsilonKGridProc; 
  
  double complex *Gr = gwParam->Gr;
  double complex *Gl = gwParam->Gl;
  double complex *Gg = gwParam->Gg;
  double complex *Pr = gwParam->Pr;
  double complex *Pl = gwParam->Pl;
  double complex *Pg = gwParam->Pg;
  double complex *Wr = gwParam->Wr;
  double complex *Wl = gwParam->Wl;
  double complex *Wg = gwParam->Wg;
  double complex *Sr = gwParam->Sr;
  double complex *Sl = gwParam->Sl;
  double complex *Sg = gwParam->Sg;


  // 1. Generate Gl, Gg from ImGr spectural function
  for(iGrid=0;iGrid<numGridProcK;iGrid++){
    #pragma omp parallel for private(jGrid,index)
    for(jGrid=0;jGrid<numGridW;jGrid++){
      index = iGrid*numGridW+jGrid;
      Gl[index] = 2*I*cimag(Gr[index])*thetaW_mu[jGrid];
      Gg[index] = -2*I*cimag(Gr[index])*(1.0-thetaW_mu[jGrid]);
    }
  }

  // 2. FFT Gl Gg k,w->k,t
  fftKWtoRT(gwParam, Gl);
  fftKWtoRT(gwParam, Gg);

  
  // 4. Calculate Pl, Pg, Pr
  for(iGrid=0;iGrid<numGridProcR;iGrid++){
    #pragma omp parallel for private(jGrid,index)
    for(jGrid=0;jGrid<numGridT;jGrid++){
      index = iGrid*numGridT+jGrid;
      Pl[index] = 2*I*Gl[index]*conj(Gg[index]);
      //Pg[index] = 2*I*Gg[index]*conj(Gl[index]);
      Pg[index] = -conj(Pl[index]);
      Pr[index] = -alpha_rs*M_PI*(Pg[index]-Pl[index])*thetaT[jGrid];
    }   
  }
  
  // 5. Pr FFT r,t->k,w
  fftRTtoKW(gwParam, Pr);
  //printf
  if(myid==0){
     for(jGrid=0;jGrid<numGridT;jGrid++){
        printf("%f %f %f\n", wGrid[jGrid],creal(Gr[jGrid]),cimag(Gr[jGrid]));
     }
  } 
  // 6. Calculate Wr,Wl,Wg from Pr
  for(iGrid=0;iGrid<numGridProcK;iGrid++){
    #pragma omp parallel for private(jGrid,index)
    for(jGrid=0;jGrid<numGridW;jGrid++){
      index = iGrid*numGridW+jGrid;
      Wr[index] = vkGridProc[iGrid]/(1.0-vkGridProc[iGrid]*(creal(Pr[index])+I*fabs(cimag(Pr[index]))));
      Wl[index] = 2.0*I*fabs(cimag(Wr[index]))*thetaW[jGrid];
      Wg[index] = 2.0*I*fabs(cimag(Wr[index]))*(1.0-thetaW[jGrid]);
    }
  }

  // 7. Wl,Wg FFT k,w->r,t
  fftKWtoRT(gwParam, Wl);
  fftKWtoRT(gwParam, Wg);

  // 8. Calculate Sl,Sg,Sr
  for(iGrid=0;iGrid<numGridProcR;iGrid++){
    #pragma omp parallel for private(jGrid,index)
     for(jGrid=0;jGrid<numGridT;jGrid++){
       index = iGrid*numGridT+jGrid;
       Sl[index] = I*Gl[index]*Wl[index];
       Sg[index] = I*Gg[index]*Wg[index];
       Sr[index] = (Sg[index] - Sl[index])*thetaT[jGrid];
     }
   }

  // 9. Sr FFT r,t->k,w
  fftRTtoKW(gwParam, Sr);
  
  // 10. Update Gr
  for(iGrid=0;iGrid<numGridProcK;iGrid++){
    #pragma omp parallel for private(jGrid,index)
    for(jGrid=0;jGrid<numGridW;jGrid++){
      index = iGrid*numGridW+jGrid;
//      Gr[index] = 1.0/(wGrid[jGrid]-epsilonKGridProc[iGrid]-mu-(alpha_rs*(-creal(Sr[index])+I*cimag(Sr[index]))+Sr_HF[iGrid]));
      Gr[index] = 1.0/(wGrid[jGrid]-epsilonKGridProc[iGrid]-mu-(alpha_rs*(-creal(Sr[index])+I*cimag(Sr[index]))));
    }
  }  
}

// FFT r,t->k,w
void fftRTtoKW(GWPARAM *gwParam, double complex *input){
  int iGrid,jGrid,kGrid,iProc;
  int index;
  int numGridK = gwParam->numGridK;
  int numGridR = gwParam->numGridR;
  int numGridT = gwParam->numGridT;
  int numGridW = gwParam->numGridW;
  int numGridProcK = gwParam->numGridProcK;
  int numGridProcR = gwParam->numGridProcR;
  int numGridProcT = gwParam->numGridProcT;
  int numGridProcW = gwParam->numGridProcW;
  int numProc = gwParam->numProc;
  int myid = gwParam->myid;
  int countRK = gwParam->countRK;
  int countTW = gwParam->countTW;
  int iThread;
  int numThreads = gwParam->numThreads;
  int *allCountsRK = gwParam->allCountsRK;
  int *allCountsTW = gwParam->allCountsTW;
  int *displsRK = gwParam->displsRK;
  int *displsTW = gwParam->displsTW;
  int div,res;

  double complex *temp,*inputFake;
  MKL_Complex16 alpha;
  //double complex alpha = 1.0;

  alpha.real = 1.0;
  alpha.imag = 0.0;

  if(numProc>1){
    temp = (double complex*)malloc(numGridR*numGridT*sizeof(double complex));
    inputFake = (double complex*)malloc(numGridProcW*numGridR*sizeof(double complex));
  }
  else{
    inputFake = input;
  }

  fftw_complex **in_backward_rk = gwParam->in_backward_rk;
  fftw_complex **in_backward_tw = gwParam->in_backward_tw;

  // 1. Transfrom t->w (r,t->r,w)
  #pragma omp parallel private(iThread,iGrid)
  {
    iThread = omp_get_thread_num();
    #pragma omp for
    for(iGrid=0;iGrid<numGridProcR;iGrid++){
      memcpy(in_backward_tw[iThread],&input[iGrid*numGridT],numGridT*sizeof(double complex));
      fftw_execute(gwParam->plan_tw_backward[iThread]);
      memcpy(&input[iGrid*numGridT],in_backward_tw[iThread],numGridT*sizeof(double complex));
    }
  }

  // 2. possible MPI gatherv
  if(numProc>1){
    MPI_Gatherv(input,countRK,MPI_DOUBLE,temp,allCountsRK,displsRK,
                MPI_DOUBLE,0,MPI_COMM_WORLD);
  }

  // 3. transpose and scatterv (r,w->w,r)
  if(numProc==1){
    mkl_zimatcopy('c','t',numGridR,numGridT,alpha,input,numGridR,numGridT);
  }
  else{
    if(myid==0){
      mkl_zimatcopy('c','t',numGridR,numGridT,alpha,temp,numGridR,numGridT);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatterv(temp,allCountsTW,displsTW,MPI_DOUBLE,inputFake,countTW,
                 MPI_DOUBLE,0,MPI_COMM_WORLD);
  }

  // inputFake multiply by rGrid*dt*numGridT
  for(iGrid=0;iGrid<numGridProcW;iGrid++){
    #pragma omp parallel for private(jGrid,index)
    for(jGrid=0;jGrid<numGridK;jGrid++){
      index = iGrid*numGridK+jGrid;
      inputFake[index] *=gwParam->rGrid[jGrid]*gwParam->dt*numGridT; 
    }
  }
  // inputFake symmetrize
  for(iGrid=0;iGrid<numGridProcW;iGrid++){
    #pragma omp parallel for private(jGrid,index)
    for(jGrid=numGridK/2+1;jGrid<numGridK;jGrid++){
      index = iGrid*numGridK+jGrid;
      inputFake[index] = inputFake[iGrid*numGridK+numGridK-jGrid];
    }
  }

  // 4. Transform r->k (w,r->w,k)
  #pragma omp parallel private(iThread,iGrid)
  {
    iThread = omp_get_thread_num();
    #pragma omp for
    for(iGrid=0;iGrid<numGridProcW;iGrid++){
      memcpy(in_backward_rk[iThread],&inputFake[iGrid*numGridT],numGridR*sizeof(double complex));
      fftw_execute(gwParam->plan_rk_backward[iThread]);
      memcpy(&inputFake[iGrid*numGridT],in_backward_rk[iThread],numGridR*sizeof(double complex));
    }
  }

  // inputFake multiply by -1j/kGrid*dr*numGridR/(2.0*pi)**2 * (2.0*pi)**3
  for(iGrid=0;iGrid<numGridProcW;iGrid++){
    #pragma omp parallel for private(jGrid,index)
    for(jGrid=0;jGrid<numGridK;jGrid++){
      index = iGrid*numGridK+jGrid;
      inputFake[index] *= -I*2.0*M_PI/gwParam->kGrid[jGrid]*gwParam->dr*numGridR; 
    }
  }

  // 5. MPI gatherv 
  if(numProc>1){
    MPI_Gatherv(inputFake,countTW,MPI_DOUBLE,temp,allCountsTW,displsTW,
                MPI_DOUBLE,0,MPI_COMM_WORLD);
  }

  // 6. transpose and scatterv (w,k->k,w)
  if(numProc==1){
    mkl_zimatcopy('c','t',numGridT,numGridR,alpha,input,numGridT,numGridR);
  }
  else{
    if(myid==0){
      mkl_zimatcopy('c','t',numGridT,numGridR,alpha,temp,numGridT,numGridR);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatterv(temp,allCountsRK,displsRK,MPI_DOUBLE,input,countRK,
                 MPI_DOUBLE,0,MPI_COMM_WORLD);
  }

  if(numProc>1){
    free(temp);
    free(inputFake);
  }  
}


// FFT k,w->r,t
void fftKWtoRT(GWPARAM *gwParam, double complex *input){
  int iGrid,jGrid,kGrid,iProc;
  int index;
  int iThread;
  int numGridK = gwParam->numGridK;
  int numGridR = gwParam->numGridR;
  int numGridT = gwParam->numGridT;
  int numGridW = gwParam->numGridW;
  int numGridProcK = gwParam->numGridProcK;
  int numGridProcR = gwParam->numGridProcR;
  int numGridProcT = gwParam->numGridProcT;
  int numGridProcW = gwParam->numGridProcW;
  int numProc = gwParam->numProc;
  int myid = gwParam->myid;
  int countRK = gwParam->countRK;
  int countTW = gwParam->countTW;
  int *allCountsRK = gwParam->allCountsRK;
  int *allCountsTW = gwParam->allCountsTW;
  int *displsRK = gwParam->displsRK;
  int *displsTW = gwParam->displsTW;
  int div,res;

  double complex *temp,*inputFake;
  MKL_Complex16 alpha;
  //double complex alpha = 1.0;

  alpha.real = 1.0;
  alpha.imag = 0.0;


  if(numProc>1){
    temp = (double complex*)malloc(numGridR*numGridT*sizeof(double complex));
    inputFake = (double complex*)malloc(numGridProcW*numGridR*sizeof(double complex));
  }
  else{
    inputFake = input;
  }

  fftw_complex **in_forward_rk = gwParam->in_forward_rk;
  fftw_complex **in_forward_tw = gwParam->in_forward_tw;

  // 1. Transfrom w->t (k,w->k,t)
  #pragma omp parallel private(iThread,iGrid)
  {
    iThread = omp_get_thread_num();
    #pragma omp for
    for(iGrid=0;iGrid<numGridProcR;iGrid++){
      memcpy(in_forward_tw[iThread],&input[iGrid*numGridT],numGridT*sizeof(double complex));
      fftw_execute(gwParam->plan_tw_forward[iThread]);
      memcpy(&input[iGrid*numGridT],in_forward_tw[iThread],numGridT*sizeof(double complex));
    }
  }

  // 2. possible MPI gatherv
  if(numProc>1){
    MPI_Gatherv(input,countRK,MPI_DOUBLE,temp,allCountsRK,displsRK,
                MPI_DOUBLE,0,MPI_COMM_WORLD);
  }

  // 3. transpose and scatterv (k,t->t,k)
  if(numProc==1){
    mkl_zimatcopy('c','t',numGridR,numGridT,alpha,input,numGridR,numGridT);
  }
  else{
    if(myid==0){
      mkl_zimatcopy('c','t',numGridR,numGridT,alpha,temp,numGridR,numGridT);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatterv(temp,allCountsTW,displsTW,MPI_DOUBLE,inputFake,countTW,
                 MPI_DOUBLE,0,MPI_COMM_WORLD);
  }

  // inputFake multiply by kGrid/dt/numGridT
  for(iGrid=0;iGrid<numGridProcW;iGrid++){
    #pragma omp parallel for private(jGrid,index)
    for(jGrid=0;jGrid<numGridK;jGrid++){
      index = iGrid*numGridK+jGrid;
      inputFake[index] *= gwParam->kGrid[jGrid]/gwParam->dt/numGridT; 
    }
  }
  // inputFake symmetrize
  for(iGrid=0;iGrid<numGridProcW;iGrid++){
    #pragma omp parallel for private(jGrid,index)
    for(jGrid=numGridK/2+1;jGrid<numGridK;jGrid++){
      index = iGrid*numGridK+jGrid;
      inputFake[index] = inputFake[iGrid*numGridK+numGridK-jGrid];
    }
  }

  // 4. Transform k->r (t,k->t,r)
  #pragma omp parallel private(iThread,iGrid)
  {
    iThread = omp_get_thread_num();
    #pragma omp for
    for(iGrid=0;iGrid<numGridProcW;iGrid++){
      memcpy(in_forward_rk[iThread],&inputFake[iGrid*numGridT],numGridR*sizeof(double complex));
      fftw_execute(gwParam->plan_rk_forward[iThread]);
      memcpy(&inputFake[iGrid*numGridT],in_forward_rk[iThread],numGridR*sizeof(double complex));
    }
  }

  // inputFake multiply by 1j*2*pi/(2*pi)**2/rGrid/dr/numGridR
  for(iGrid=0;iGrid<numGridProcW;iGrid++){
    #pragma omp parallel for private(jGrid,index)
    for(jGrid=0;jGrid<numGridK;jGrid++){
      index = iGrid*numGridK+jGrid;
      inputFake[index] *= I/2.0/M_PI/gwParam->rGrid[jGrid]/gwParam->dr/numGridR; 
     }
  }
  // 5. MPI gatherv 
  if(numProc>1){
    MPI_Gatherv(inputFake,countTW,MPI_DOUBLE,temp,allCountsTW,displsTW,
                MPI_DOUBLE,0,MPI_COMM_WORLD);
  }

  // 6. transpose and scatterv (t,r->r,t)
  if(numProc==1){
    mkl_zimatcopy('c','t',numGridT,numGridR,alpha,input,numGridT,numGridR);
  }
  else{
    if(myid==0){
      mkl_zimatcopy('c','t',numGridT,numGridR,alpha,temp,numGridT,numGridR);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatterv(temp,allCountsRK,displsRK,MPI_DOUBLE,input,countRK,
                 MPI_DOUBLE,0,MPI_COMM_WORLD);
  }

  if(numProc>1){
    free(temp);
    free(inputFake);
  }  
}


// Initialize
void init(GWPARAM *gwParam){
  int numProc = gwParam->numProc;
  int myid = gwParam->myid;
  int iProc;
  int div,res;
  int iGrid,jGrid,index;
  int numGridK,numGridR,numGridT,numGridW;
  int numGridProcK,numGridProcR,numGridProcT,numGridProcW;
  int displKRPoint; // starting index of Vk, epsilonK on each process
  int numThreads,iThread;
  int *allCountsRK,*allCountsTW,*displsRK,*displsTW;
  double w,dw,k,dk,dr,mu;
  double epsilonF;

  gwParam->numGridK = NGRID_RK;
  gwParam->numGridR = NGRID_RK;
  gwParam->numGridT = NGRID_TW;
  gwParam->numGridW = NGRID_TW;
  numGridK = gwParam->numGridK;
  numGridR = gwParam->numGridR;
  numGridT = gwParam->numGridT;
  numGridW = gwParam->numGridW;

  if(numProc>1){
    div = numGridW/numProc;
    res = numGridW%numProc;
    if(myid<res)numGridProcW = (div+1);
    else numGridProcW = div;
    numGridProcT = numGridProcW;
    div = numGridR/numProc;
    res = numGridR%numProc;
    if(myid<res){
      numGridProcR = (div+1);
      displKRPoint = (div+1)*myid; 
    }   
    else{
      numGridProcR = div;
      displKRPoint = (div+1)*res+(myid-res)*div;
    }   
    numGridProcK = numGridProcR;
  }
  else{
    displKRPoint = 0;
    numGridProcW = numGridW;
    numGridProcT = numGridT;
    numGridProcK = numGridK;
    numGridProcR = numGridR;
  }
  gwParam->numGridProcW = numGridProcW;
  gwParam->numGridProcT = numGridProcT;
  gwParam->numGridProcR = numGridProcR;
  gwParam->numGridProcK = numGridProcK;

  gwParam->Gr = (double complex*)malloc(numGridProcR*numGridT*sizeof(double complex));
  gwParam->Gl = (double complex*)malloc(numGridProcR*numGridT*sizeof(double complex));
  gwParam->Gg = (double complex*)malloc(numGridProcR*numGridT*sizeof(double complex));
  gwParam->Pr = (double complex*)malloc(numGridProcR*numGridT*sizeof(double complex));
  gwParam->Pl = (double complex*)malloc(numGridProcR*numGridT*sizeof(double complex));
  gwParam->Pg = (double complex*)malloc(numGridProcR*numGridT*sizeof(double complex));
  gwParam->Wr = (double complex*)malloc(numGridProcR*numGridT*sizeof(double complex));
  gwParam->Wl = (double complex*)malloc(numGridProcR*numGridT*sizeof(double complex));
  gwParam->Wg = (double complex*)malloc(numGridProcR*numGridT*sizeof(double complex));
  gwParam->Sr = (double complex*)malloc(numGridProcR*numGridT*sizeof(double complex));
  gwParam->Sl = (double complex*)malloc(numGridProcR*numGridT*sizeof(double complex));
  gwParam->Sg = (double complex*)malloc(numGridProcR*numGridT*sizeof(double complex));

  gwParam->countRK = 2*numGridProcR*numGridT;
  gwParam->countTW = 2*numGridProcT*numGridR;

  gwParam->allCountsRK = (int*)calloc(numProc,sizeof(int));
  gwParam->allCountsTW = (int*)calloc(numProc,sizeof(int));
  gwParam->displsRK = (int*)calloc(numProc,sizeof(int));
  gwParam->displsTW = (int*)calloc(numProc,sizeof(int));
  allCountsRK = gwParam->allCountsRK;
  allCountsTW = gwParam->allCountsTW;
  displsRK = gwParam->displsRK;
  displsTW = gwParam->displsTW;

  if(numProc>1){
    for(iProc=0;iProc<numProc;iProc++){
      div = numGridW/numProc;
      res = numGridW%numProc;
      if(iProc<res){
        allCountsTW[iProc] = (div+1)*numGridR*2;
      }
      else{
        allCountsTW[iProc] = div*numGridR*2;
      }
    }//endfor iProc
    displsTW[0] = 0;
    for(iProc=1;iProc<numProc;iProc++){
      displsTW[iProc] = displsTW[iProc-1]+allCountsTW[iProc];
    }
    for(iProc=0;iProc<numProc;iProc++){
      div = numGridR/numProc;
      res = numGridR%numProc;
      if(iProc<res){
        allCountsRK[iProc] = (div+1)*numGridT*2;
      }
      else{
        allCountsRK[iProc] = div*numGridT*2;
      }
    }//endfor iProc
    displsRK[0] = 0;
    for(iProc=1;iProc<numProc;iProc++){
      displsRK[iProc] = displsRK[iProc-1]+allCountsRK[iProc];
    }
  }//endif numProc

  // initialize openMP
  // You can set the number of threads by setting OMP_NUM_THREADS
  gwParam->numThreads = omp_get_num_threads();
  numThreads = gwParam->numThreads;

  // FFTW initialize
  
  gwParam->in_forward_rk = (fftw_complex**)malloc(numThreads*sizeof(fftw_complex*));
  gwParam->in_backward_rk = (fftw_complex**)malloc(numThreads*sizeof(fftw_complex*));
  gwParam->in_forward_tw = (fftw_complex**)malloc(numThreads*sizeof(fftw_complex*));
  gwParam->in_backward_tw = (fftw_complex**)malloc(numThreads*sizeof(fftw_complex*));

  for(iThread=0;iThread<numThreads;iThread++){
    gwParam->in_forward_rk[iThread] = (fftw_complex*)fftw_malloc(numGridR*sizeof(fftw_complex));
    gwParam->in_backward_rk[iThread] = (fftw_complex*)fftw_malloc(numGridR*sizeof(fftw_complex));
    gwParam->in_forward_tw[iThread] = (fftw_complex*)fftw_malloc(numGridT*sizeof(fftw_complex));
    gwParam->in_backward_tw[iThread] = (fftw_complex*)fftw_malloc(numGridT*sizeof(fftw_complex));
  }
  gwParam->plan_rk_forward = (fftw_plan*)malloc(numThreads*sizeof(fftw_plan)); 
  gwParam->plan_rk_backward = (fftw_plan*)malloc(numThreads*sizeof(fftw_plan));
  gwParam->plan_tw_forward = (fftw_plan*)malloc(numThreads*sizeof(fftw_plan));
  gwParam->plan_tw_backward = (fftw_plan*)malloc(numThreads*sizeof(fftw_plan));
  for(iThread=0;iThread<numThreads;iThread++){
    gwParam->plan_rk_forward[iThread] = fftw_plan_dft_1d(numGridR,gwParam->in_forward_rk[iThread],
                                     gwParam->in_forward_rk[iThread],FFTW_FORWARD,FFTW_MEASURE);
    gwParam->plan_rk_backward[iThread] = fftw_plan_dft_1d(numGridR,gwParam->in_backward_rk[iThread],
                                     gwParam->in_backward_rk[iThread],FFTW_BACKWARD,FFTW_MEASURE);
    gwParam->plan_tw_forward[iThread] = fftw_plan_dft_1d(numGridT,gwParam->in_forward_tw[iThread],
                                     gwParam->in_forward_tw[iThread],FFTW_FORWARD,FFTW_MEASURE);
    gwParam->plan_tw_backward[iThread] = fftw_plan_dft_1d(numGridT,gwParam->in_forward_tw[iThread],
                                     gwParam->in_forward_tw[iThread],FFTW_BACKWARD,FFTW_MEASURE);
  }

  /*
  gwParam->plan_rk_forward = fftw_plan_dft_1d(numGridR,gwParam->in_forward_rk,
                                     gwParam->in_forward_rk,FFTW_FORWARD,FFTW_MEASURE);
  gwParam->plan_rk_backward = fftw_plan_dft_1d(numGridR,gwParam->in_backward_rk,
                                     gwParam->in_backward_rk,FFTW_BACKWARD,FFTW_MEASURE);
  gwParam->plan_tw_forward = fftw_plan_dft_1d(numGridT,gwParam->in_forward_tw,
                                     gwParam->in_forward_tw,FFTW_FORWARD,FFTW_MEASURE);
  gwParam->plan_tw_backward = fftw_plan_dft_1d(numGridT,gwParam->in_forward_tw,
                                     gwParam->in_forward_tw,FFTW_BACKWARD,FFTW_MEASURE);
  */

  // Initialize grid spacing
  gwParam->dk = 2.0*K_MAX/NGRID_RK; // -K_MAX to K_MAX
  gwParam->dw = 2.0*W_MAX/NGRID_TW; // -W_MAX to W_MAX
  gwParam->dr = 2.0*M_PI/(NGRID_RK*gwParam->dk);
  gwParam->dt = 2.0*M_PI/(NGRID_TW*gwParam->dw);
  gwParam->mu = 0.0;
  gwParam->epsilonF = EPSILON_F;
  dw = gwParam->dw;
  dk = gwParam->dk;
  dr = gwParam->dr;
  mu = gwParam->mu;
  epsilonF = gwParam->epsilonF;


  // Initialize more grids
  double *wGrid;
  double *tGrid;
  double *kGrid;
  double *rGrid;
  double *thetaW_mu;
  double *thetaW;
  double *thetaT;

  double *vkGrid,*epsilonKGrid;
  double *vkGridProc;
  double *epsilonKGridProc; 

  gwParam->wGrid = (double*)calloc(numGridW,sizeof(double));
  gwParam->tGrid = (double*)calloc(numGridW,sizeof(double));
  gwParam->kGrid = (double*)calloc(numGridK,sizeof(double));
  gwParam->rGrid = (double*)calloc(numGridR,sizeof(double));
  gwParam->thetaW_mu = (double*)calloc(numGridW,sizeof(double));
  gwParam->thetaW = (double*)calloc(numGridW,sizeof(double));
  gwParam->thetaT = (double*)calloc(numGridW,sizeof(double));
  wGrid = gwParam->wGrid;
  tGrid = gwParam->tGrid;
  kGrid = gwParam->kGrid;
  rGrid = gwParam->rGrid;
  thetaW_mu = gwParam->thetaW_mu;
  thetaW = gwParam->thetaW;
  thetaT = gwParam->thetaT;

  gwParam->vkGridProc = (double*)calloc(numGridProcK,sizeof(double));
  gwParam->epsilonKGridProc = (double*)calloc(numGridProcK,sizeof(double));
  vkGridProc = gwParam->vkGridProc;
  epsilonKGridProc = gwParam->epsilonKGridProc;
  vkGrid = (double*)calloc(numGridK,sizeof(double));
  epsilonKGrid = (double*)calloc(numGridK,sizeof(double));
  
  // thetaW, thetaT,thetaW_mu, tGrid, wGrid
  for(iGrid=0;iGrid<numGridW;iGrid++){
    if(iGrid<numGridW/2+1){
      w = iGrid*dw;
      thetaW[iGrid] = 1.0;
      thetaT[iGrid] = 1.0;
      tGrid[iGrid] = iGrid*gwParam->dt;
    }  
    else{
      w = (iGrid-numGridW)*dw;
      tGrid[iGrid] = (iGrid-numGridW)*gwParam->dt;
    }   
    if(w>=mu)thetaW_mu[iGrid] = 1.0;
    wGrid[iGrid] = w;
  }

  // kGrid, rGrid 
  kGrid[0]= K_ETA;
  rGrid[0]= R_ETA;
  for(iGrid=1;iGrid<numGridK;iGrid++){
    if(iGrid<numGridK/2+1){
      kGrid[iGrid] = iGrid*dk;
      rGrid[iGrid] = iGrid*dr;
    }  
    else{
      kGrid[iGrid] = (iGrid-numGridK)*dk;
      rGrid[iGrid] = (iGrid-numGridK)*dr;
    }   
  }

  // vkGridProc, epsilonKGridProc 
  vkGrid[0] = 4.0*M_PI/(K_ETA*K_ETA);
  epsilonKGrid[0] = -epsilonF;
  for(iGrid=1;iGrid<numGridK;iGrid++){
    if(iGrid<numGridK/2+1){
      k = iGrid*dk;
    }
    else{
      k = (iGrid-numGridK)*dk;
    }
    vkGrid[iGrid] = 4.0*M_PI/(k*k);
    epsilonKGrid[iGrid] = k*k-epsilonF;
  }
  memcpy(vkGridProc,&vkGrid[displKRPoint],numGridProcR*sizeof(double));
  memcpy(epsilonKGridProc,&epsilonKGrid[displKRPoint],numGridProcR*sizeof(double));

  // Initialize prefactor (unit conversion) 
  gwParam->alpha_rs = ALPHA_RS;

  // initialize Gr 
  double complex *Gr = gwParam->Gr;
  for(iGrid=0;iGrid<numGridProcK;iGrid++){
    #pragma omp parallel for private(jGrid,index)
    for(jGrid=0;jGrid<numGridW;jGrid++){
      index = iGrid*numGridW+jGrid;
      Gr[index] = 1.0/(wGrid[jGrid]-epsilonKGridProc[iGrid]+I*GF_ETA);
    }
  }

  free(vkGrid);
  free(epsilonKGrid);

}


void clean(GWPARAM *gwParam){
  int numThreads = gwParam->numThreads;
  int iThread; 
  
  for(iThread=0;iThread<numThreads;iThread++){
    fftw_destroy_plan(gwParam->plan_rk_forward[iThread]); 
    fftw_destroy_plan(gwParam->plan_rk_backward[iThread]);
    fftw_destroy_plan(gwParam->plan_tw_forward[iThread]);
    fftw_destroy_plan(gwParam->plan_tw_backward[iThread]);
    fftw_free(gwParam->in_forward_rk[iThread]);
    fftw_free(gwParam->in_backward_rk[iThread]);
    fftw_free(gwParam->in_forward_tw[iThread]);
    fftw_free(gwParam->in_backward_tw[iThread]);
  }
  free(gwParam->plan_rk_forward);
  free(gwParam->plan_rk_backward);
  free(gwParam->plan_tw_forward);
  free(gwParam->plan_tw_backward);
  free(gwParam->in_forward_rk);
  free(gwParam->in_backward_rk);
  free(gwParam->in_forward_tw);
  free(gwParam->in_backward_tw);
}





