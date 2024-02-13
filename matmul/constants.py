from manim import *

FONT_TITLE = "Linux Biolinum"

TEX_GENERAL = TexTemplate(
    tex_compiler="xelatex",
    output_format=".xdv",
    documentclass="\\documentclass[preview]{standalone}",
    preamble="\\usepackage{libertine}[newtxmath]\n\\usepackage{amsmath}",
)

NAIVE_GEMM = """ \
__global__ void naive_gemm(
    float* A, float* B, float* C,
    uint32_t M, uint32_t N, uint32_t K
) {
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M or col >= N)
        return;

    float sum = 0;
    for (uint32_t k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
"""

OPTIMIZED_GEMM_0 = """ \
__global__ void optimized_gemm_v0(
    float* A, float* B, float* C,
    uint32_t M, uint32_t N, uint32_t K
) {
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    auto tx = threadIdx.x, ty = threadIdx.y;
    auto row = blockIdx.y * blockDim.y + ty;
    auto col = blockIdx.x * blockDim.x + tx;
    if (row >= M or col >= N)
        return;

    float sum = 0;
    for (uint32_t k = 0; k < K / BLOCK_SIZE; ++k) {
        shared_A[ty][tx] = A[row * K + k * BLOCK_SIZE + tx];
        shared_B[ty][tx] = B[(k * BLOCK_SIZE + ty) * N + col];
        __syncthreads();

        for (uint32_t i = 0; i < BLOCK_SIZE; ++i) {
            sum += shared_A[ty][i] * shared_B[i][tx];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}
"""
