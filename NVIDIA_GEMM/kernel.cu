#include "kernel_4.cuh"
#include "kernel_5.cuh"
#include "kernel_2.cuh"
#include "kernel_3.cuh"
#include<iostream>
#include<cuda_runtime.h>
#include "driver_types.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#define M 16384 
#define K 16384
#define N 16384
#define BLOCK_SIZE 32
using namespace std;
void initial(float* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = (float)(rand() % 100 + 1);
    }
}
float runCublasMul(float* array_A, float* array_B, float* array_C_coblas) {
    float* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
    cudaMalloc((void**)&d_A, sizeof(float) * M * K);
    cudaMalloc((void**)&d_B, sizeof(float) * K * N);
    cudaMalloc((void**)&d_C, sizeof(float) * M * N);

    cudaMemcpy(d_A, array_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, array_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1;
    float beta = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //C=A*B
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,                    //矩阵B的列数
        M,                    //矩阵A的行数
        K,                    //公共维度
        &alpha,
        d_B,                  //矩阵B的指针
        N,                    //矩阵B实际上以行优先存储，所以主维度为列数
        d_A,                  //矩阵A的指针
        K,                    //矩阵A实际上以行优先存储，所以主维度为列数
        &beta,
        d_C,
        N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(array_C_coblas, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return time;
}

float runKernel_2(float* A, float* B, float* C) {
    float* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
    cudaMalloc((void**)&d_A, sizeof(float) * M * K);
    cudaMalloc((void**)&d_B, sizeof(float) * K * N);
    cudaMalloc((void**)&d_C, sizeof(float) * M * N);

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N - 1) / dimBlock.x + 1, (M - 1) / dimBlock.y + 1);
    cudaEventRecord(start);
    Shared << <dimGrid, dimBlock >> > (d_A, d_B, d_C, M, K, K, N, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return time;
}

float runKernel_3(float* A, float* B, float* C) {
    float* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
    cudaMalloc((void**)&d_A, sizeof(float) * M * K);
    cudaMalloc((void**)&d_B, sizeof(float) * K * N);
    cudaMalloc((void**)&d_C, sizeof(float) * M * N);

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE / 2);
    dim3 dimGrid((N - 1) / dimBlock.x + 1, (M - 1) / (dimBlock.y * 2) + 1);
    cudaEventRecord(start);
    sharedILPkernel << <dimGrid, dimBlock >> > (d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return time;
}

float runKernel_4(float* A, float* B, float* C) {
    float* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
    cudaMalloc((void**)&d_A, sizeof(float) * M * K);
    cudaMalloc((void**)&d_B, sizeof(float) * K * N);
    cudaMalloc((void**)&d_C, sizeof(float) * M * N);

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    dim3 blockDim(256);      //线程块大小为 (BM*BN)/(TM*TN) = (128*128)/(8*8)=256)
    dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
    cudaEventRecord(start);
    doubleBuffering<128, 128, 8, 8, 8> << <gridDim, blockDim >> > (M, N, K, 1, d_A, d_B, 0, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return time;
}

float runKernel_5(float* A, float* B, float* C) {
    float* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
    cudaMalloc((void**)&d_A, sizeof(float) * M * K);
    cudaMalloc((void**)&d_B, sizeof(float) * K * N);
    cudaMalloc((void**)&d_C, sizeof(float) * M * N);

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    const int NUM_THREADS = 128;
    const int BN = 128;
    const int BM = 128;
    const int BK = 16;
    const int WN = 64;
    const int WM = 64;
    const int WNITER = 4;   //WMITER=1;
    const int TN = 4;
    const int TM = 8;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    dim3 blockDim(NUM_THREADS);
    dim3 gridDim(CEIL_DIV(N, N), CEIL_DIV(M, BM));
    cudaEventRecord(start);
    sgemmWarptiling<BM, BN, BK, WM, WN, WNITER, TM,
        TN, NUM_THREADS>
        << <gridDim, blockDim >> > (M, N, K, 1, d_A, d_B, 0, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return time;
}


void checkResult(float* array_A, float* array_B, int size) {
    double epsilon = 1.0E-8;
    for (int i = 0; i < size; i++) {
        if (abs(array_A[i] - array_B[i]) > epsilon) {
            printf("Error! Matrix[%05d]:%0.8f != %0.8f\n", i, array_A[i], array_B[i]);
            return;
        }
    }
    printf("Check result success!\n");
}

int main() {
    float* array_A, * array_B, * array_C_2, * array_C_3, * array_C_4, * array_C_5, * array_C_cublas;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);
    array_A = (float*)malloc(size_A);
    array_B = (float*)malloc(size_B);
    array_C_2 = (float*)malloc(size_C);
    array_C_3 = (float*)malloc(size_C);
    array_C_4 = (float*)malloc(size_C);
    array_C_5 = (float*)malloc(size_C);
    array_C_cublas = (float*)malloc(size_C);
    initial(array_A, M * K);
    initial(array_B, K * N);
    float time1 = runCublasMul(array_A, array_B, array_C_cublas);
    cout << "Cublas Solution: " << time1 << "ms" << endl;
    float time2 = runKernel_2(array_A, array_B, array_C_2);
    cout << "Shared Memory Solution: " << time2 << "ms" << endl;
    cout << time1 / time2 * 100 << "%" << endl;
    checkResult(array_C_cublas, array_C_2, M * N);
    float time3 = runKernel_3(array_A, array_B, array_C_3);
    cout << "Cublas Solution: " << time1 << "ms" << endl;
    cout << "Register Optimize Solution: " << time3 << "ms" << endl;
    cout << time1 / time3 * 100 << "%" << endl;
    checkResult(array_C_cublas, array_C_3, M * N);
    float time4 = runKernel_4(array_A, array_B, array_C_4);
    cout << "Cublas Solution: " << time1 << "ms" << endl;
    cout << "Double Buffering Solution: " << time4 << "ms" << endl;
    cout << time1 / time4 * 100 << "%" << endl;
    checkResult(array_C_cublas, array_C_4, M * N);
    float time5 = runKernel_5(array_A, array_B, array_C_5);
    cout << "Cublas Solution: " << time1 << "ms" << endl;
    cout << "Warptiling Solution: " << time5 << "ms" << endl;
    cout << time1 / time5 * 100 << "%" << endl;
    checkResult(array_C_cublas, array_C_5, M * N);
    free(array_A);
    free(array_B);
    free(array_C_2);
    free(array_C_3);
    free(array_C_4);
    free(array_C_5);
    free(array_C_cublas);
    return 0;
}