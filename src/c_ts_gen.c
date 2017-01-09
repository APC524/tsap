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
void c_ar1_gen(double * array, const double rho, const double sigma, const int time_, const int num, const  int burnin){
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

void c_ma1_gen(double * array, const double rho, const double constant, const int time_, const int num, const int burnin ){
	/* simulate data from MA1 model
	X_t = constant + e_t + \rho * e_{t-1}, where e_t is a sequence of white noise*/

	int T = time_ + burnin;
	int index = 0;
	double white_noise[T];
	for( int i =0; i<num; ++i){
		// first generate a sequnce of white noise
		for (int j=0; j<T; ++j){
			white_noise[j] = randn();
		}
		for( int j=0; j< T; ++j){
			if( j==0 ){
					array[index] = white_noise[0] + constant;
			}
			else{
				array[index] = constant + white_noise[j] +rho * white_noise[j-1];
			}
			++index;
		}
	}
}


void c_arch1_gen(double *array, double a0, double a1, int time_, int num, int burnin){
	/* simulate data from arch model
	X_t = e_t * sqrt{ h_t}
	h_t = a_0 + a_1 * X_{t-1}^2
	*/

	int T = time_ + burnin;
	int index = 0;
	double temp;
	for ( int i =0; i < num; i++){
		for(int j =0; j < T; j++){
			if (j ==0){
					array[index] = sqrt(a0) * randn();
			}
			else{
				temp = a1 * array[index-1] * array[index-1] +a0;
				array[index] = sqrt(temp) * randn();
			}
		}
	}

}




void c_garch11_gen(double * array, double a, double b, double c, const int time_, const int num, const int burnin ){

	/* simulate data from garch(1,1) model
	X_t = e_t * h_t,
	h_t = a + b * X_{t-1}^2 + c * h_{t-1}
	*/

	int T = time_ + burnin;
	int index = 0;
	double temp[T] =0;
	for ( int i =0; i < num; i++){
		for(int j =0; j < T; j++){
			temp[j] = 0;
		}

		for(int j =0; j < T; j++){
			if (j ==0){
					temp[j] = a;
					array[index] = sqrt( temp[j])* randn();
			}
			else{
				temp[j] = a + b * array[index-1] * array[index-1] + c * temp[j-1];
				array[index] = sqrt(temp[j]) * randn();
			}
		}
	}


}


/*  simulate data from ARMA model */

void c_arma_gen(double * array, double* ar, int p, const double * ma, const int q, const double sigma, const int time_, const int num, const int burnin ){

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
            temp = temp + ma[k] * noise[j -k-1];
          }
        }
        array[index] = temp;
      }
      ++index;
    }
  }


}



void c_garch_gen(double * array, double * constant, double* garch, int p, double * arch, int q,double sigma, int time_, int num, int burnin ){
	/* simulate data from Garch model, which is specified by
	X_t = e_t * sqrt( h_t),
	where e_t is the standard_noise,
	and h_t = a_0 + \sum_{i=1}^q alpha_i e_{t-i}^2 + \sum _{i=1}^p \beta_i h_{t-i}
	alpha_i: Arch coefficients, beta_i: Garch coefficients

	Input:
	array: pointer to where the data is stored,
	constant: a_0,
	garch: double array with length p, coefficients of beta
	arch: double array with length q, coefficients of alpha
	*/





}
