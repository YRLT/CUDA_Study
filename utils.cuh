#include<iostream>
#include<vector>
#include<tuple>
#include <assert.h>

#include "cuda.h"
#include "cuda_runtime.h"

using namespace std;
 

const int N = 100;

struct CudaStamp {
    cudaEvent_t start;
    cudaEvent_t stop;
};


void printMatrix(float Mat[][N]);

float set_clock_end(CudaStamp* stamp, float time_elapsed);

CudaStamp* set_clock_start();

void get_device_info( );

bool is_array2D_equil(float A[][N], float B[][N]);