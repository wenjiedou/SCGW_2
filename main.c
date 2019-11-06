#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include "gw.h"


int main(int argc, char *argv[]){
  GWPARAM *gwParam = NULL;

  int numProc;
  int myid;

//  printf("I am here! haha\n");
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numProc);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

//  printf("I am here! haha\n");
  gwParam = (GWPARAM*)malloc(sizeof(GWPARAM));
  gwParam->numProc = numProc;
  gwParam->myid = myid;

//  printf("I am here! haha\n");
  init(gwParam);
  gwOneIteration(gwParam);
//  printf("I am here! haha\n");
  // calculation....

  clean(gwParam);
  printf("finish clean");
  MPI_Finalize();
  
  return 1;
}
