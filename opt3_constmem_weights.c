
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
  int constMemSize = 24 * 12 * 7 * 7; //TAs confirmed these values on piazza. They should work for any cases (7 = K for this project)
  __constant__ float weights[constMemSize];

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

  // int H_grid = H_out/TILE_WIDTH;
  int W_grid =	ceil((float)W_out/TILE_WIDTH);

  int h = (bz/W_grid)*TILE_WIDTH + tx;
  int w = (bz%W_grid)*TILE_WIDTH + ty;
  //--------------------------Pdf uses these vars (Ch 16 Page 14)

  // An example use of these macros:
  // float a = y4d(0,0,0,0)
  // y4d(0,0,0,0) = a
  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
  #define w4d(i3,i2,i1,i0) Weights[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

     if((bx < B) && (by < M) && (h < H_out) && (w < W_out)) {
        y4d(bx, by, h, w) = 0;

        for(int c = 0; c < C; c++)    //For all input feature maps (C)
          for(int p = 0; p < K; p++)  //Go through whole KxK filter
            for(int q = 0; q < K; q++)
              y4d(bx, by, h, w) += x4d(bx, c, h+p, w+q) * w4d(by, c, p, q);
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

    int W_out = W - K + 1;	//Output width (ghost elements subtracted from input width)
    int H_out = H - K + 1;	//Output height (ghost elements subtracted from input height)

    int H_grid = ceil((float)H_out/TILE_WIDTH);
    int W_grid =	ceil((float)W_out/TILE_WIDTH);

    int Z = H_grid * W_grid;

    // Set the kernel dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z);

    //Optimization 3: Copy weights into constant memory
    int weightSize = 12*7*7; //TODO: Figure out what these 2 lines mean
    if(M != 12) weightSize = weightSize * 24;
    cudaMemcpyToSymbol(weights, w.dptr_, weightSize * sizeof(float), 0, cudaMemcpyHostToDevice);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim, 0>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

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
