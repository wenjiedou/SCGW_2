#include <complex.h>
#include <fftw3.h>
#define K_ETA 1.0e-6
#define R_ETA 1.0e-6
#define GF_ETA 1.0e-3
#define EPSILON_F 0.25
#define NGRID_RK 1001
#define NGRID_TW 1001
#define K_MAX 100.0
#define W_MAX 20.0
#define ALPHA_RS 20.0
#define M_PI 3.14159265358979323846

typedef struct gwparam{
  // r: retarded l: less g: greater
  double complex *Gr,*Gl,*Gg; // Green's function G
  double complex *Pl,*Pg,*Pr; // Polarizablility P
  double complex *Wl,*Wg,*Wr; // Screen Columb W 
  double complex *Sl,*Sg,*Sr; // Self-energy Sigma
  double mu;                  // chemical potential
  double dr,dk,dt,dw;         // Grid Spacing
  double *wGrid,*tGrid,*kGrid,*rGrid; // w, t, k, R grid       
  double *thetaW_mu,*thetaT,*thetaW; // theta(w-mu), theta(T), theta(w) grid
  double *vkGridProc,*epsilonKGridProc; // vk = 4*pi/k**2, epsilon_k = k**2/2 grid 
  double alpha_rs;                // alpha*Rs 
  double epsilonF;            // epsilon_F
  int numGridK,numGridR,numGridT,numGridW; //numGridK=numGridR,numGridT=numGridW
  int numGridProcK,numGridProcR,numGridProcT,numGridProcW;
  int numGridTotal,numGridTotalProc;
  int numProc,myid;
  int countRK,countTW;        // array length in each process if RK/TW is leading dimension
  int *allCountsRK,*allCountsTW;
                              // array lengths in all processes if RK/TW are leading dimensions
  int *displsRK,*displsTW;    // displacement if RK/TW is leading dimension
  fftw_complex *in_forward_rk,*in_backward_rk,*in_forward_tw,*in_backward_tw;
  fftw_plan plan_rk_forward,plan_rk_backward,plan_tw_forward,plan_tw_backward;
}GWPARAM;

void gwOneIteration(GWPARAM*);
void fftRTtoKW(GWPARAM*, double complex*);
void fftKWtoRT(GWPARAM*, double complex*);
void init(GWPARAM*);
void clean(GWPARAM*);

//...




