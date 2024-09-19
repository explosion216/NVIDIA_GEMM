#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

//#define M 16384 
//#define K 16384
//#define N 16384
#define BLOCK_SIZE 32 
using namespace std;

// using ILP 2 to improve the performance
__global__ void sharedILPkernel(float* fpMatrixA, float* fpMatrixB,
    float* fpMatrixC, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y * 2 + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float val[2] = { 0.0f };

    __shared__ float shTileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shTileB[BLOCK_SIZE][BLOCK_SIZE];

    int iter = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int i = 0; i < iter; i++)
    {
        //read data from global memory to shared memory
        shTileA[threadIdx.y][threadIdx.x] = fpMatrixA[row * k + i * BLOCK_SIZE + threadIdx.x];
        shTileA[threadIdx.y + 16][threadIdx.x] = fpMatrixA[(row + 16) * k + i * BLOCK_SIZE + threadIdx.x];

        shTileB[threadIdx.y][threadIdx.x] = fpMatrixB[(i * BLOCK_SIZE + threadIdx.y) * n + col];
        shTileB[threadIdx.y + 16][threadIdx.x] = fpMatrixB[(i * BLOCK_SIZE + threadIdx.y + 16) * n + col];

        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++)
        {
            val[0] += shTileA[threadIdx.y][j] * shTileB[j][threadIdx.x];
            val[1] += shTileA[threadIdx.y + 16][j] * shTileB[j][threadIdx.x];
        }

        __syncthreads();
    }

    fpMatrixC[row * n + col] = val[0];
    fpMatrixC[(row + 16) * n + col] = val[1];
}
