#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 	16
#define TILE_SIZE 16
#define CUDA_MAX_NUM_THREADS 1024
#define MAT_X 32
#define MAT_Y 32
#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z
#define tx threadIdx.x
#define ty threadIdx.y

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{
__global__ void matrixMultiply(float *A, float *B, float *C, int a_row, int a_col, int b_row, int b_col, int c_row, int c_col) {
  if(a_col == b_row) {

    int Row = by*blockDim.y + ty;
    int Col = bx*blockDim.x + tx;

    if((Row < c_row) && (Col < c_col)) {

      float P = 0;

      for(int k = 0; k < a_col; ++k)
        P += A[Row*a_col+k] * B[k*b_col+Col];

       C[Row*c_col+Col] = P;
    }
  }
}

__global__ void unroll_Kernel(int B, int C, int H, int W, int K, float* x, float* X_unroll) {

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

    int c, s, h_out, w_out, h_unroll, w_unroll, w_base, p, q;
    int t = bx * CUDA_MAX_NUM_THREADS + tx;
    int W_unroll = H_out * W_out;

    if (t < (C * W_unroll)) {
        c = t / W_unroll;
        s = t % W_unroll;
        h_out = s / W_out;
        w_out = s % W_out;
        h_unroll = h_out * W_out + w_out;
        w_base = c * K * K;

        for(p = 0; p < K; p++) {
            for(q = 0; q < K; q++) {
                w_unroll = w_base + (p*K) + q;
                if(c < C && h_out+p < H && w_out + q < W/* h_unroll < W_unroll && w_unroll < K /*&& by < B && c < C && h_out+p < H_out + K && w_out + q < W_out + K*/) {
                  X_unroll[h_unroll*W_unroll + w_unroll] = x4d(0, c, h_out + p, w_out + q);
                }
            }
        }
    }

      #undef x4d
}

//
//   This function is called by new-inl.h
//   Any code you write should be executed by this function.
//   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
//
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K

    int B = x.shape_[0];
    int M = y.shape_[1];
    int C = x.shape_[1];
    int H = x.shape_[2];
    int W = x.shape_[3];
    int K = w.shape_[3];

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    /* Optimization 2: Unroll/Matrix Multiply */
    int expandedHeight =  C*K*K;  //Height of the expanded matrix,
                                  //# of input feature elems contributing to each output feature map elem
    int expandedWidth = H_out*W_out;  //Width of the expanded matrix,
                                      //# of elems in each output feature map

    // Set kernel dimensions for unroll_Kernel
    dim3 num_blocks(ceil((C * H_out * W_out) / CUDA_MAX_NUM_THREADS), 1, 1);
    dim3 num_threads(CUDA_MAX_NUM_THREADS, 1, 1);

    // Set kernel dimensions for matrixMultiply
    dim3 blocks(ceil((float)M/MAT_X), ceil((float)H_out*W_out/MAT_Y), 1);
    dim3 threads(MAT_X, MAT_Y, 1);

    // Allocate the hostC matrix
    float * X_unroll = (float *)malloc(expandedHeight*sizeof(float));

    MSHADOW_CUDA_CALL(cudaDeviceSynchronize()); // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.

    // Call the kernels
    unroll_Kernel<<<num_blocks, num_threads>>>(B, C, H, W, K, x.dptr_, X_unroll);
    matrixMultiply<<<blocks, threads>>>(w.dptr_, X_unroll, y.dptr_, M, expandedHeight, expandedHeight, expandedWidth, M, expandedWidth);

    MSHADOW_CUDA_CALL(cudaDeviceSynchronize()); // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
}

//
//    This tells mxnet how to do an op when it's not a float.
//    This is not used in the ECE408 project
//
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
