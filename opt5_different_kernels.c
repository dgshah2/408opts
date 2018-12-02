
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_


#define TILE_WIDTH 16
#define TILE_SIZE 16
#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z
#define tx threadIdx.x
#define ty threadIdx.y

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__constant__ float weights[24*12*7*7];  //TAs confirmed these values on piazza. They should work for any cases

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

  int W_out = W - K + 1;	//Output width (ghost elements subtracted from input width)
  int H_out = H - K + 1;	//Output height (ghost elements subtracted from input height)

  // An example use of these macros:
  // float a = y4d(0,0,0,0)
  // y4d(0,0,0,0) = a
  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  //Optimization 1: Tiled (shared memory) convolution
  int X_tile_width = TILE_WIDTH + K-1;
  extern __shared__ float shmem[]; //Holds both input block and filter coefficients
  float* X_shared = &shmem[0];  //Input block/input feature map
  float* K_shared = &shmem[X_tile_width*X_tile_width];  //Filter coefficients/weights

  // int H_grid = H_out/TILE_WIDTH;
  int W_grid =	ceil((float)W_out/TILE_WIDTH);	//Number of horizontal tiles per output map

  int h_base = (bz/W_grid)*TILE_SIZE;  //Vertical base out data index for the block
  int w_base = (bz%W_grid)*TILE_SIZE;  //Horizontal base out data index for the block
  int h = h_base + tx;
  int w = w_base + ty;

  float accum = 0;

  //B = # of samples in mini-batch, M = # of output feature maps

  for(int c = 0; c < C; c++) {  //For all input feature maps (C)

    //Load weights for K[by, c, ...] into shared memory
    if((tx < K) && (ty < K)) {  //Only do this within the the filter size KxK
      K_shared[tx*K + ty] = k4d(by, c, tx, ty);  //K_shared[tx, ty]
    }
    __syncthreads();

    //Load tile from X[bx, c, ...] into shared memory
    for(int i = h; i < (h_base + X_tile_width); i += TILE_WIDTH) {
      for(int j = w; j < (w_base + X_tile_width); j += TILE_WIDTH) {
          X_shared[(i-h_base)*X_tile_width + (j-w_base)] = x4d(bx, c, i, j);  //X_shared[(i-h_base), (j-w_base)]
      }
    }
    __syncthreads();

    //Accumulate output value
    for(int p = 0; p < K; p++) {	//Go through whole KxK filter
      for(int q = 0; q < K; q++) {
          accum += X_shared[(tx+p)*X_tile_width + (ty+q)] * K_shared[p*K + q]; //X_shared[(h+p), (w+q)] and K_shared[p, q]
      }
    }
    __syncthreads();
  }

  //Set output value
  if ((bx < B) && (by < M) && (h < H_out) && (w < W_out)) {
    y4d(bx, by, h, w) = accum;
  }

  #undef y4d
  #undef x4d
  #undef k4d
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
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

    int W_out = W - K + 1;
    int H_out = H - K + 1;
    int H_grid = ceil((float)H_out/TILE_WIDTH);
    int W_grid =	ceil((float)W_out/TILE_WIDTH);

    int Z = H_grid * W_grid;


    // Set the kernel dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z);

    //Used for Optimization 1
    int X_tile_width = TILE_WIDTH + K-1;
    size_t shmem_size = sizeof(float) * ((X_tile_width*X_tile_width) + K*K);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
