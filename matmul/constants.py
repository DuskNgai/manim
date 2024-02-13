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
// optimized_gemm_v0
__shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
__shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

auto tx = threadIdx.x, ty = threadIdx.y;
auto row = blockIdx.y * blockDim.y + ty;
auto col = blockIdx.x * blockDim.x + tx;

float sum = 0.0f;
for (uint32_t k = 0; k < div_round_up(K, BLOCK_SIZE); ++k) {
    uint32_t col_A = k * BLOCK_SIZE + tx;
    uint32_t row_B = k * BLOCK_SIZE + ty;
    shared_A[ty][tx] = col_A < K ? A[row * K + col_A] : 0.0f;
    shared_B[ty][tx] = row_B < K ? A[row_B * N + col] : 0.0f;
    __syncthreads();

    for (uint32_t i = 0; i < BLOCK_SIZE; ++i) {
        sum += shared_A[ty][i] * shared_B[i][tx];
    }
    __syncthreads();
}
if (row < M and col < N) {
    C[row * N + col] = sum;
}
"""
