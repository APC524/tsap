/*
ts_gen.c

generate data from various time series models,
the data is stored in a two dimensional array.

*/

#include "stdlib.h"
#include "math.h"
#include "limits.h"

/*
 the following c code to generate gaussian distribution is obtained from Wikipedia https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform?oldformat=true
*/


double randn()
{
	const double epsilon = 1e-30;
	const double two_pi = 2.0*3.14159265358979323846;

	static double z0, z1;

	double u1, u2;
	do
	 {
	   u1 = rand() * (1.0 / RAND_MAX);
	   u2 = rand() * (1.0 / RAND_MAX);
	 }
	while ( u1 <= epsilon );

	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0;
}

/* a simple AR(1) model */
void c_ar1_gen(double * array, double rho, double sigma, int time_, int num, int burnin){
  /*
  simulate data from AR(1) model: y_t = \rho * y_{t-1} + e_t, where
  e_t is Gaussian with variance sigma^2,
  rho is the parameter,

  input:
  array: pointer to where the data is stored,
  rho: parameter of AR(1),
  sigma: standard error of the noise term,
  time: time of the time series
  burnin: burnin period
  num: number of independent trials
  */

    int i, j;
    int T = time_+ burnin;
    int index = 0;
    for(i=0; i < num; ++i){
      for(j=0; j< T; ++j){
        if(j ==0) {
          array[index] = sigma * randn();
        }
        else{
          array[index] = array[index-1] * rho + sigma * randn();
        }
        ++index;
      }
    }
}



/*  simulate data from ARMA model */

void c_arma_gen(double * array, double* ar, int p, double * ma, int q,double sigma, int time_, int num, int burnin ){

  /*
  simulate data from one dimensional ARMA model:

          y_t =  \sum _{i=1}^p phi_i y_{t-i} +\sum_{i=1}^q \theta_i e_{t-i} + e_t, where
  e_k is Gaussian with variance sigma^2,
  \phi_1, ... \phi_p are the autoregressive parametrs
  \theta_1, ..., \theta_q are the moving-average parameters


  input:
  array: pointer to where the data is stored,
  ar: parameter of AR model,
  p: order of AR

  ma: parameter of MA model,
  q: order of MA

  sigma: standard error of the noise term,
  time_: time of the time series
  burnin: burnin period
  num: number of independent trials
  */
  int i, j;
  int T = time_+ burnin;
  int index = 0;
  double temp;

  double noise[T];
  for(i=0; i < num; ++i){
    for(int k =0; k<T; ++k){
      noise[k] = sigma * randn();
    }

    for(j=0; j< T; ++j){
      if( j ==0){
        array[index] = noise[j];
      }
      else{
        temp = noise[j];
        for(int k =0; k < j; ++k){
          if(k < p){
            temp = temp + ar[k] * array[index-1-k];
          }
          if(k < q){
            temp = temp + ma[k] * noise[j -k-1]
          }
        }
        array[index] = temp;
      }
      ++index;
    }
  }


}
