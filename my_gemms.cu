#include<stdio.h>

#include<vector>
#include<iostream>

#include<windows.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "device_launch_parameters.h"

#include "utils.cuh"

using namespace std;

float run_cublas_gemm(int dev, 
    const float A[][N],
    const float B[][N],
    float C[][N]) {
    cublasHandle_t handle;

    float alpha, beta;

    alpha = 1.f;    beta = 0.f;

    // create cuBLAS handle
    cublasCreate(&handle);
    cudaError_t cudastat = cudaSetDevice(dev);
    
    float* pCublasA = nullptr;
    float* pCublasB = nullptr;
    float* pCublasC = nullptr;

    cudastat = cudaMalloc((void**)&pCublasA, N * N * sizeof(float));
    cudastat = cudaMalloc((void**)&pCublasB, N * N * sizeof(float));
    cudastat = cudaMalloc((void**)&pCublasC, N * N * sizeof(float));

    cudaMemcpy(pCublasA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pCublasB, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    float time_elapsed = 0;
    CudaStamp *cublas_stamp = set_clock_start();

    // Gemm
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
        N, N, N, &alpha, pCublasA, N, pCublasB, N, &beta, pCublasC, N);

    
    time_elapsed = set_clock_end(cublas_stamp, time_elapsed);
    
    cudaDeviceSynchronize();
    
    delete cublas_stamp;

    cudaMemcpy(C, pCublasC, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    

    cublasDestroy(handle);
    cudaFree(pCublasA);    cudaFree(pCublasB);    cudaFree(pCublasC);

    return time_elapsed;
 
}

__global__ void my_tiling_gemm(
    float* dev_a,
    float* dev_b,
    float* dev_c) {
    // dims: block( N / 16, N / 4), threads(32, 2)

    __shared__ float B_vecs[N][4]; // bandwidth: 128bit
    float thread_res[4] = {0,0,0,0};
    __shared__ float res[16][16]; // 16 row per block

    int col_begin = blockIdx.y * 4;
    int row_begin = blockIdx.x * 16;

    // read B tile in SM by warp
    int B_tile_nums = N / 16;
    int row_in_Btile = threadIdx.y * 8 + threadIdx.x / 4;
    int col_in_Btile = threadIdx.x % 4;
    for (int tile = 0; tile < B_tile_nums; tile++) {
        B_vecs[tile * 16 + row_in_Btile][col_in_Btile] = dev_a[(tile * 16 + row_in_Btile) * N + col_begin + col_in_Btile];
    }

    __syncthreads();

    // mm with reading A_vec_tile, 4 threads per row
    int A_tile_nums = N / 4;
    int row_in_Atile = threadIdx.y * 8 + threadIdx.x / 4;
    int col_in_Atile = threadIdx.x % 4;
    for (int tile = 0; tile < A_tile_nums; tile++) {
        thread_res[0] += dev_a[row_begin * N + (tile * 4 + col_in_Atile)] * B_vecs[(tile * 4 + col_in_Atile)][row_begin];
    }

    res[row_in_Atile][col_in_Atile * 4] = thread_res[0];
    res[row_in_Atile][col_in_Atile * 4 + 1] = thread_res[1];
    res[row_in_Atile][col_in_Atile * 4 + 2] = thread_res[2];
    res[row_in_Atile][col_in_Atile * 4 + 3] = thread_res[4];

    __syncthreads();
    // reduce result by single thread and save in first 4 cols in SM
    for (int i = 4; i < 16; i += 4) {
        res[row_in_Atile][col_in_Atile] += res[row_in_Atile][col_in_Atile + i];
    }

    dev_c[(row_begin + row_in_Atile) * N + col_begin + col_in_Atile] = res[row_in_Atile][col_in_Atile];

    return;
}

__global__ void my_simple_gemm(
    float* dev_a,
    float* dev_b,
    float* dev_c,
    int threads_size
){
    const float co_i = 1.0 / N;

    int threadid = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
    float res = 0.0;
    int i = 0, j = 0;

    int idx = threadid;

    for (; idx < N * N;) {
        i = floor(idx * co_i);
        j = idx - i * N;
        for (int k = 0; k < N; k++) {
             res += dev_a[i * N + k] * dev_b[k * N + j];
        }
        dev_c[j * N + i] = res;
        idx = idx + threads_size;
    }
    
    return ;
}

 float run_my_gemm(int dev,
     const float A[][N],
     const float B[][N],
     float C[][N]) {
     
     cudaError_t cudastat = cudaSetDevice(dev);

     float* pCudaA = nullptr;
     float* pCudaB = nullptr;
     float* pCudaC = nullptr;

     cudaError_t cudaStatus;

     cudaStatus = cudaMalloc((void**)&pCudaA, N * N * sizeof(float));
     cudaStatus = cudaMalloc((void**)&pCudaB, N * N * sizeof(float));
     cudaStatus = cudaMalloc((void**)&pCudaC, N * N * sizeof(float));
 
     cudaStatus = cudaMemcpy(pCudaA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);

     if (pCudaA == NULL) {
         printf("couldn't allocate pCudaA GPU memory in my Gemm\n");
         return -1;
     }

     cudaStatus = cudaMemcpy(pCudaB, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

     if (pCudaB == NULL) {
         printf("couldn't allocate pCudaB GPU memory in my Gemm\n");
         return  -1;
     }

     float time_elapsed = 0;
     CudaStamp* cublas_stamp = set_clock_start();


     // for(i,j,k) {C[i][j] += A[i][k] * B[k][j]} 
     int s = 0;

     dim3 grid(N/16, N/4);
     dim3 block(32, 2);

     my_tiling_gemm << <grid, block >> > (pCudaA, pCudaB, pCudaC);

     //if (N * N < 1024 * 6 * 2) {
     //    int threads = floor(N * N /(2 * 6));
     //    dim3 grid(2, 6);
     //    dim3 block(threads);
     //    my_simple_gemm << <grid, block >> > (pCudaA, pCudaB, pCudaC, 2 * 6 * threads);
     //}
     //else{
     //    dim3 grid(2, 6);
     //    dim3 block(1024);
     //    my_simple_gemm << <grid, block >> > (pCudaA, pCudaB, pCudaC, 2 * 6 * 1024);
     //}


     time_elapsed = set_clock_end(cublas_stamp, time_elapsed);

     cudaDeviceSynchronize();


     delete cublas_stamp;

     cudaMemcpy(C, pCudaC, N * N * sizeof(float), cudaMemcpyDeviceToHost);

     cudaFree(pCudaA); cudaFree(pCudaB); cudaFree(pCudaC);
     return time_elapsed;
}

int main() {
	int dev = 0;
    int iter = 15;

    cout << "test N = " << N << ", iter = " << iter << endl;

    int size = sizeof(float) * N * N;
 
    float (*A)[N] = new float[N][N];
    float (*B)[N] = new float[N][N];
    float (*C)[N] = new float[N][N];

    srand(2019);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() / float(2019);
            B[i][j] = rand() / float(2019);
            C[i][j] = 0;
        }
    }

    float avg_cublas_time = 0;

    for (int it = 0; it < iter; it++) {
        avg_cublas_time += run_cublas_gemm(dev, A, B, C);
        cudaDeviceSynchronize();
    }
    cout << avg_cublas_time / iter << " --- cublas time " << endl;

    Sleep(1000);

    float(*my_C)[N] = new float[N][N];
    float my_gemm_time = 0;
    
    for (int it = 0; it < iter; it++) {
        my_gemm_time += run_my_gemm(dev, A, B, my_C);;
        cudaDeviceSynchronize();
    }
    cout << my_gemm_time / iter << " --- my gemm time " << endl;

    //std::cout << "C:\n" << endl;
    //printMatrix(C);
    //std::cout << "myC:\n" << endl;
    //printMatrix(my_C);


    cout << "check result: " << is_array2D_equil(C, my_C) << endl;

    delete[] A;
    delete[] B;
    delete[] C;

    cout << "freed A, B, C points" << endl;

    //get_device_info();

	return 0;
}