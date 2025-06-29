#include "matrix_lib.h"
#include <cuda_runtime.h>
#include <stdio.h>

/**
 * @brief Kernel CUDA que realiza a multiplicação escalar em um vetor de entrada.
 *
 * @param scalar Valor escalar a ser multiplicado.
 * @param input Vetor de entrada na memória da GPU.
 * @param output Vetor de saída na memória da GPU.
 * @param size Tamanho total do vetor (número de elementos).
 */
__global__ void scalar_mult_kernel(float scalar, float *input, float *output, int size) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float *in_ptr = input + global_idx;
    float *out_ptr = output + global_idx;

    for (int idx = global_idx; idx < size; idx += stride) {
        *out_ptr = scalar * (*in_ptr);
        in_ptr += stride;
        out_ptr += stride;
    }
}

/**
 * @brief Função host que realiza a multiplicação de uma matriz por um escalar usando CUDA.
 *
 * @param scalar_value Valor escalar a ser multiplicado.
 * @param m Ponteiro para a matriz de entrada.
 * @param r Ponteiro para a matriz de saída (resultado).
 * @return 0 em caso de sucesso, ou código de erro negativo.
 *
 * @retval -1 Erro de validação ou alocação.
 */
int scalar_matrix_mult(float scalar_value, matrix *m, matrix *r) {
    if (!m || !r || !m->values || !r->values || m->rows != r->rows || m->cols != r->cols) {
        return -1;
    }

    int total_size = m->rows * m->cols;
    float *deviceInput = NULL, *deviceOutput = NULL;

    cudaError_t err;
    err = cudaMalloc((void**)&deviceInput, total_size * sizeof(float));
    if (err != cudaSuccess) {
        return -1;
    }
    err = cudaMalloc((void**)&deviceOutput, total_size * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(deviceInput);
        return -1;
    }

    err = cudaMemcpy(deviceInput, m->values, total_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        return -1;
    }

    scalar_mult_kernel<<<blocksPerGrid, threadsPerBlock>>>(scalar_value, deviceInput, deviceOutput, total_size);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro no kernel: %s\n", cudaGetErrorString(err));
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        return -1;
    }

    err = cudaMemcpy(r->values, deviceOutput, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        return -1;
    }

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return 0;
}

/**
 * @brief Kernel CUDA que realiza multiplicação de matrizes usando indexação 1D.
 *
 * @param mA Matriz A (dimensões m x n) na GPU.
 * @param mB Matriz B (dimensões n x p) na GPU.
 * @param mC Matriz resultado C (dimensões m x p) na GPU.
 * @param m Número de linhas da matriz A.
 * @param n Número de colunas da matriz A e número de linhas da matriz B.
 * @param p Número de colunas da matriz B.
 */
__global__ void matrix_mult_1d(float *mA, float *mB, float *mC, int m, int n, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < m * p; i += stride) {
        int row = i / p;
        int col = i % p;

        float sum = 0.0f;

        float* m1_ptr = mA + row * n;
        float* m2_ptr = mB + col;

        for (int k = 0; k < n; k++) {
            sum += *(m1_ptr + k) * *(m2_ptr + k * p);
        }

        mC[i] = sum;
    }
}

/**
 * @brief Função host que realiza a multiplicação de duas matrizes usando CUDA.
 *
 * @param m1 Ponteiro para a matriz A (dimensões m x n).
 * @param m2 Ponteiro para a matriz B (dimensões n x p).
 * @param r Ponteiro para a matriz resultado (dimensões m x p).
 * @return 0 em caso de sucesso, ou código de erro negativo.
 *
 * @retval -1 Ponteiros nulos ou inválidos.
 * @retval -2 Dimensões incompatíveis para multiplicação.
 * @retval -3 Erro de alocação na GPU.
 * @retval -4 Erro ao copiar dados para ou da GPU.
 * @retval -5 Erro durante a execução do kernel.
 */
int matrix_matrix_mult(matrix *m1, matrix *m2, matrix *r) {
    if (!m1 || !m2 || !r || !m1->values || !m2->values || !r->values)
        return -1;

    if (m1->cols != m2->rows)
        return -2;

    int m = m1->rows;
    int n = m1->cols;
    int p = m2->cols;

    memset(r->values, 0, sizeof(float) * m * p);

    size_t sizeA = m * n * sizeof(float);
    size_t sizeB = n * p * sizeof(float);
    size_t sizeC = m * p * sizeof(float);

    float* deviceA = NULL; 
    float* deviceB = NULL; 
    float* deviceC = NULL;

    cudaError_t err;

    err = cudaMalloc((void **)&deviceA, sizeA);
    if (err != cudaSuccess) return -3;

    err = cudaMalloc((void **)&deviceB, sizeB);
    if (err != cudaSuccess) {
        cudaFree(deviceA);
        return -3;
    }

    err = cudaMalloc((void **)&deviceC, sizeC);
    if (err != cudaSuccess) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        return -3;
    }

    err = cudaMemcpy(deviceA, m1->values, sizeA, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
        return -4;
    }

    err = cudaMemcpy(deviceB, m2->values, sizeB, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
        return -4;
    }

    matrix_mult_1d<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, m, n, p);
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro no kernel: %s\n", cudaGetErrorString(err));
        cudaFree(deviceA); 
        cudaFree(deviceB); 
        cudaFree(deviceC);
        return -5;
    }

    err = cudaMemcpy(r->values, deviceC, sizeC, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
        return -4;
    }

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}
