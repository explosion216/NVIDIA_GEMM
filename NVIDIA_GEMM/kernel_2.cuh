#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

#define M 16384 
#define K 16384
#define N 16384
#define BLOCK_SIZE 32 
using namespace std;

__global__ void Shared(float* A, float* B, float* C, int numARows,    //核心思想是将矩阵 A 和 B 分为大小为 BLOCK_SIZE x BLOCK_SIZE 的局部块，利用共享内存存储这些局部数据，然后利用线程计算局部矩阵乘法并累加结果。
    int numAColumns, int numBRows,
    int numBColumns, int numCRows,
    int numCColumns) {
    __shared__ float ds_M[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float ds_N[BLOCK_SIZE][BLOCK_SIZE];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * BLOCK_SIZE + ty;
    int Col = bx * BLOCK_SIZE + tx;
    float Pvalue = 0;
    for (int m = 0; m < (numAColumns - 1) / BLOCK_SIZE + 1; ++m) {
        if (Row < numARows && m * BLOCK_SIZE + tx < numAColumns) {  //将矩阵 A 的局部数据加载到共享内存数组 ds_M。如果当前线程所负责的元素位于矩阵 A 的有效范围内，则从矩阵 A 中读取对应元素；否则，将 ds_M 中对应位置设为 0.0。
            ds_M[ty][tx] = A[Row * numAColumns + m * BLOCK_SIZE + tx];
        }
        else {
            ds_M[ty][tx] = 0.0;
        }
        if (Col < numBColumns && m * BLOCK_SIZE + ty < numBRows) {
            ds_N[ty][tx] = B[(m * BLOCK_SIZE + ty) * numBColumns + Col];
        }
        else {
            ds_N[ty][tx] = 0.0;
        }
        __syncthreads();    //同步线程，确保所有线程已完成从矩阵 A 和 B 中加载数据到共享内存。
        for (int k = 0; k < BLOCK_SIZE; ++k) {  //内层循环执行局部矩阵乘法。遍历当前线程负责的共享内存 ds_M 的一行和 ds_N 的一列，执行点乘并将结果累加到 Pvalue。
            Pvalue += ds_M[ty][k] * ds_N[k][tx];
        }
        __syncthreads();    //再次同步线程，确保所有线程已完成局部矩阵乘法，以便进行下一轮迭代。
    }
    if (Row < numCRows && Col < numCColumns) {
        C[Row * numCColumns + Col] = Pvalue;
    }
}
