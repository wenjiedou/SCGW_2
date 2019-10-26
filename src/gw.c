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

  double mu = gwParam->mu;
  double w,k;
  double dw = gwParam->dw;
  double pre1 = gwParam->pre1;
  double pre2 = gwParam->pre2;
  double epsilonF = gwParam->epsilonF;

  double *wGrid = gwParam->wGrid;
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
    for(jGrid=0;jGrid<numGridT;jGrid++){
      index = iGrid*numGridT+jGrid;
      Pl[index] = 2*I*Gl[index]*conj(Gg[index]);
      //Pg[index] = 2*I*Gg[index]*conj(Gl[index]);
      Pg[index] = -conj(Pl[index]);
      Pr[index] = -pre1*(Pg[index]-Pl[index])*thetaT[jGrid];
    }   
  }
  
  // 5. Pr FFT r,t->k,w
  fftRTtoKW(gwParam, Pr);
  
  // 6. Calculate Wr,Wl,Wg from Pr
  for(iGrid=0;iGrid<numGridProcK;iGrid++){
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
    for(jGrid=0;jGrid<numGridW;jGrid++){
      index = iGrid*numGridW+jGrid;
//      Gr[index] = 1.0/(wGrid[jGrid]-epsilonKGridProc[iGrid]-mu-(pre2*(-creal(Sr[index])+I*cimag(Sr[index]))+Sr_HF[iGrid]));
      Gr[index] = 1.0/(wGrid[jGrid]-epsilonKGridProc[iGrid]-mu-(pre2*(-creal(Sr[index])+I*cimag(Sr[index]))));
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

  fftw_complex *in_backward_rk = gwParam->in_backward_rk;
  fftw_complex *in_backward_tw = gwParam->in_backward_tw;

  // 1. Transfrom t->w (r,t->r,w)
  for(iGrid=0;iGrid<numGridProcR;iGrid++){
    memcpy(in_backward_tw,&input[iGrid*numGridT],numGridT*sizeof(double complex));
    fftw_execute(gwParam->plan_tw_backward);
    memcpy(&input[iGrid*numGridT],in_backward_tw,numGridT*sizeof(double complex));
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

  // 4. Transform r->k (w,r->w,k)
  for(iGrid=0;iGrid<numGridProcW;iGrid++){
    memcpy(in_backward_rk,&inputFake[iGrid*numGridT],numGridR*sizeof(double complex));
    fftw_execute(gwParam->plan_rk_backward);
    memcpy(&inputFake[iGrid*numGridT],in_backward_rk,numGridR*sizeof(double complex));
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

  fftw_complex *in_forward_rk = gwParam->in_forward_rk;
  fftw_complex *in_forward_tw = gwParam->in_forward_tw;

  // 1. Transfrom w->t (k,w->k,t)
  for(iGrid=0;iGrid<numGridProcR;iGrid++){
    memcpy(in_forward_tw,&input[iGrid*numGridT],numGridT*sizeof(double complex));
    fftw_execute(gwParam->plan_tw_forward);
    memcpy(&input[iGrid*numGridT],in_forward_tw,numGridT*sizeof(double complex));
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

  // 4. Transform k->r (t,k->t,r)
  for(iGrid=0;iGrid<numGridProcW;iGrid++){
    memcpy(in_forward_rk,&inputFake[iGrid*numGridT],numGridR*sizeof(double complex));
    fftw_execute(gwParam->plan_rk_forward);
    memcpy(&inputFake[iGrid*numGridT],in_forward_rk,numGridR*sizeof(double complex));
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
  int *allCountsRK,*allCountsTW,*displsRK,*displsTW;
  double w,dw,k,dk,mu;
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

  // FFTW initialize
  gwParam->in_forward_rk = (fftw_complex*)fftw_malloc(numGridR*sizeof(fftw_complex));
  gwParam->in_backward_rk = (fftw_complex*)fftw_malloc(numGridR*sizeof(fftw_complex));
  gwParam->in_forward_tw = (fftw_complex*)fftw_malloc(numGridT*sizeof(fftw_complex));
  gwParam->in_backward_tw = (fftw_complex*)fftw_malloc(numGridT*sizeof(fftw_complex));
  
  gwParam->plan_rk_forward = fftw_plan_dft_1d(numGridR,gwParam->in_forward_rk,
                                     gwParam->in_forward_rk,FFTW_FORWARD,FFTW_MEASURE);
  gwParam->plan_rk_backward = fftw_plan_dft_1d(numGridR,gwParam->in_backward_rk,
                                     gwParam->in_backward_rk,FFTW_BACKWARD,FFTW_MEASURE);
  gwParam->plan_tw_forward = fftw_plan_dft_1d(numGridT,gwParam->in_forward_tw,
                                     gwParam->in_forward_tw,FFTW_FORWARD,FFTW_MEASURE);
  gwParam->plan_tw_backward = fftw_plan_dft_1d(numGridT,gwParam->in_forward_tw,
                                     gwParam->in_forward_tw,FFTW_BACKWARD,FFTW_MEASURE);

  // Initialize grid spacing
  gwParam->dk = 2.0*K_MAX/NGRID_RK; // -K_MAX to K_MAX
  gwParam->dw = 2.0*W_MAX/NGRID_TW; // -W_MAX to W_MAX
  gwParam->dr = 2.0*M_PI/(NGRID_RK*gwParam->dk);
  gwParam->dt = 2.0*M_PI/(NGRID_TW*gwParam->dw);
  gwParam->mu = 0.0;
  gwParam->epsilonF = EPSILON_F;
  dw = gwParam->dw;
  dk = gwParam->dk;
  mu = gwParam->mu;
  epsilonF = gwParam->epsilonF;


  // Initialize more grids
  double *wGrid;
  double *kGrid;
  double *rGrid;
  double *thetaW_mu;
  double *thetaW;
  double *thetaT;

  double *vkGrid,*epsilonKGrid;
  double *vkGridProc;
  double *epsilonKGridProc; 

  gwParam->wGrid = (double*)calloc(numGridW,sizeof(double));
  gwParam->kGrid = (double*)calloc(numGridK,sizeof(double));
  gwParam->rGrid = (double*)calloc(numGridR,sizeof(double));
  gwParam->thetaW_mu = (double*)calloc(numGridW,sizeof(double));
  gwParam->thetaW = (double*)calloc(numGridW,sizeof(double));
  gwParam->thetaT = (double*)calloc(numGridW,sizeof(double));
  wGrid = gwParam->wGrid;
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

  for(iGrid=0;iGrid<numGridW;iGrid++){
    if(iGrid<numGridW/2){
      w = iGrid*dw;
      thetaW[iGrid] = 1.0;
      thetaT[iGrid] = 1.0;
    }  
    else{
      w = (iGrid-numGridW)*dw;
    }   
    if(w>=mu)thetaW_mu[iGrid] = 1.0;
    wGrid[iGrid] = w;
  }

  for(iGrid=0;iGrid<numGridW;iGrid++){
    if(iGrid<numGridW/2){
      w = iGrid*dw;
      thetaW[iGrid] = 1.0;
      thetaT[iGrid] = 1.0;
    }  
    else{
      w = (iGrid-numGridW)*dw;
    }   
    if(w>=mu)thetaW_mu[iGrid] = 1.0;
    wGrid[iGrid] = w;
  }


  vkGrid[0] = 4.0*M_PI/(K_ETA*K_ETA);
  epsilonKGrid[0] = -epsilonF;
  for(iGrid=1;iGrid<numGridK;iGrid++){
    if(iGrid<numGridK/2){
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

  // initialize Gr 
  double complex *Gr = gwParam->Gr;
  for(iGrid=0;iGrid<numGridProcK;iGrid++){
    for(jGrid=0;jGrid<numGridW;jGrid++){
      index = iGrid*numGridW+jGrid;
      Gr[index] = 1.0/(wGrid[jGrid]-epsilonKGridProc[iGrid]+I*GF_ETA);
    }
  }

  free(vkGrid);
  free(epsilonKGrid);
}


void clean(GWPARAM *gwParam){
  
  fftw_destroy_plan(gwParam->plan_rk_forward); 
  fftw_destroy_plan(gwParam->plan_rk_backward);
  fftw_destroy_plan(gwParam->plan_tw_forward);
  fftw_destroy_plan(gwParam->plan_tw_backward);
  fftw_free(gwParam->in_forward_rk);
  fftw_free(gwParam->in_backward_rk);
  fftw_free(gwParam->in_forward_tw);
  fftw_free(gwParam->in_backward_tw);

}




