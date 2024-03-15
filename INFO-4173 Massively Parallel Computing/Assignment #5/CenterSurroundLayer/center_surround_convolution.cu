#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>
#include <vector>

// Surrounding computation in the forward pass
template <typename scalar_t>
__device__ scalar_t S(const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>& I,
                      const int b, const int c_i, const int Y, const int X)
{
    // TODO Equation 2
	scalar_t sum = (scalar_t)0.0f;
	for (int y = Y - 1; y <= Y + 1; y++) {
	    for (int x = X - 1; x <= X + 1; x++) {
	        if (!((y == Y) && (x == X))) sum += I[b][c_i][y][x];
	    }
	}
	return sum;  // function returns scalar_t
}

/** forward kernel
 * c)
 * - the forward pass of the Center Surround Convolution (forward pass form Equation 1)
 * - The kernel takes I, w c , w s, w b and O in form of
 *    torch::PackedTensorAccessor32<scalar t> objects and writes into O
 */
template <typename scalar_t>
__global__ void center_surround_convolution_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> I,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> w_c,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> w_s,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> w_b,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> O)
{
    // index
    const int x_o = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_o = blockIdx.y * blockDim.y + threadIdx.y;

    // Access sizes of tensor
    const int N = O.size(0);
    const int C_i = I.size(1);
    const int C_o = O.size(1);
    const int Y_o = O.size(2);
    const int X_o = O.size(3);

    // Check for the valid region
    if (y_o >= Y_o || x_o >= X_o)
        return;

    // TODO Implement Equation 1 here.
    // One has to do the calculation for each batch and
    // output element
    // Hint: Do not forget the bias w_b

    // IMPORTANT: Each thread has to compute N x C_o x 1 elements
    for (int b = 0; b < N; b++) {                   // batch
            for (int c_o = 0; c_o < C_o; c_o++) {       // output channel
                scalar_t sum = (scalar_t)0;
                for (int c_i = 0; c_i < C_i; c_i++) {
                    sum += I[b][c_i][y_o + 1][x_o + 1] * w_c[c_i][c_o] + S(I, b, c_i, y_o + 1, x_o + 1) * w_s[c_i][c_o];
                }
                O[b][c_o][y_o][x_o] = sum + w_b[c_o];
            }
        }
}

/** backward kernels
 * d) Implement the backward pass of the Center Surround Convolution.
 * - Write CUDA kernels to compute the partial derivatives dL_dI, dL_dw_c,
 *   dL_dw_s and dL_dw_b:
 *     - dl_dw_kernel handles Equation 4 and 5.
 *     - dl_dw_b_kernel handles equation 3.
 *     - dl_dI_kernel handles Equation 6.
 */
template <typename scalar_t>
__global__ void
dL_dw_kernel(const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dO,
             const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> I,
             torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dL_dw_c,
             torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dL_dw_s)
{
    // index
    const int c_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int c_o = blockIdx.y * blockDim.y + threadIdx.y;

    const int N = I.size(0);
    const int C_i = I.size(1);
    const int C_o = dL_dO.size(1);
    const int Y_o = dL_dO.size(2);
    const int X_o = dL_dO.size(3);

    if (c_i >= C_i || c_o >= C_o)
        return;

    // Write results to w_c_result and w_s_result
    scalar_t sum_w_c = (scalar_t)0;
    scalar_t sum_w_s = (scalar_t)0;
    // TODO compute dL_dw_c and dL_dw_s here (Equation 4, and 5)

    // TODO: LOOPS
    // All batches can be accumulated in sum_w_c/sum_w_s
    // Remember dL_dO is already given

    for(int b=0; b < N; b++) {                                  // batch
        for(int y_o = 0; y_o < Y_o; y_o++) {                    // output height
            for(int x_o = 0; x_o < X_o; x_o++) {                // output width
                sum_w_c += dL_dO[b][c_o][y_o][x_o] * I[b][c_i][y_o + 1][x_o + 1];
                sum_w_s += dL_dO[b][c_o][y_o][x_o] * S(I, b, c_i, y_o + 1, x_o + 1);
            }
        }
    }

    // Output
    dL_dw_c[c_i][c_o] = sum_w_c;
    dL_dw_s[c_i][c_o] = sum_w_s;
}

// dL_dw_b
template <typename scalar_t>
__global__ void
dL_dw_b_kernel(const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dO,
               torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dL_dw_b)
{
    const int c_o = blockIdx.x * blockDim.x + threadIdx.x;

    const int N = dL_dO.size(0);
    const int C_o = dL_dO.size(1);
    const int Y_o = dL_dO.size(2);
    const int X_o = dL_dO.size(3);

    if (c_o >= C_o)
        return;

    scalar_t sum = (scalar_t)0;

    // TODO compute dL_dw_b here (Equation 3)

    // TODO: LOOPS
    // All batches can be accumulated in sum, for all
    // y_o in 0..Y_o and x_0 in 0..X_o
    // Remember dL_dO is already given
    // IMPORTANT: Each thread computes 1 element: dL_dw_b[c_0]
   for(int b = 0; b < N; b++) {                             // batch
       for(int y_o = 0; y_o < Y_o; y_o++) {                 // output height
           for(int x_o = 0; x_o < X_o; x_o++) {             // output width
               sum += dL_dO[b][c_o][y_o][x_o];
           }
       }
   }

   dL_dw_b[c_o] = sum;
}

// dL_dI (Equation 6)
template <typename scalar_t>
__global__ void dL_dI_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dO_padded,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> w_c,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> w_s,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dL_dI)
{
    const int x_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_i = blockIdx.y * blockDim.y + threadIdx.y;

    const int N = dL_dO_padded.size(0);
    const int C_i = dL_dI.size(1);
    const int C_o = dL_dO_padded.size(1);
    const int Y_i = dL_dI.size(2);
    const int X_i = dL_dI.size(3);

    if (y_i >= Y_i || x_i >= X_i)
        return;

    // TODO implement Equation 6 here
    // dl_dO_padded is already given. Just lookup the correct value
    // One has to do the calculation for each batch and input element
    // IMPORTANT: Each thread computes N x C_i x 1 elements (N = batch size, C_i = number of channels in the input Tensor)
    for (int b = 0; b < N; b++) {                   // batch
        for (int c_i = 0; c_i < C_i; c_i++) {       // input channel
            scalar_t sum = (scalar_t)0;
            for (int c_o = 0; c_o < C_o; c_o++) {
                sum += dL_dO_padded[b][c_o][y_i + 1][x_i + 1] * w_c[c_i][c_o] + S(dL_dO_padded, b, c_o, y_i + 1, x_i + 1) * w_s[c_i][c_o];
            }
            dL_dI[b][c_i][y_i][x_i] = sum;
        }
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
    CHECK_CUDA(x);                                                                                 \
    CHECK_CONTIGUOUS(x)

/**  A c++ function that allocates memory for O and calls the kernel
 *   with appropriate block- and grid dimensions. This function takes I,
 *   w_c, w_s and w_b as torch::Tensor objects and returns the computed O tensor
 *   as std::vector<torch::Tensor>.
 */
std::vector<torch::Tensor> center_surrond_convolution_forward(torch::Tensor I, torch::Tensor w_c,
                                                              torch::Tensor w_s, torch::Tensor w_b)
{
    // Check inputs
    CHECK_INPUT(I);
    CHECK_INPUT(w_c);
    CHECK_INPUT(w_s);
    CHECK_INPUT(w_b);

    // Allocate Memory for O
    // - Extract the correct sizes first
    const auto N = I.size(0); // Batch size
    const auto C_o = w_c.size(1); // Output size
    const auto H = I.size(2);
    const auto W = I.size(3);

    auto O = torch::empty({N, C_o, H - 2, W - 2}, I.options());

    // Call the kernel (only for floating types)
    const dim3 block_dim(32, 32);
    const dim3 grid_dim((W - 3) / 32 + 1, (H - 3) / 32 + 1);

    // This launches the kernel
    AT_DISPATCH_FLOATING_TYPES(
        I.scalar_type(), "center_surround_convolution_cuda_forward",
        (
            [&]
            {
                center_surround_convolution_forward_kernel<scalar_t><<<grid_dim, block_dim>>>(
                    I.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    w_c.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    w_s.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    w_b.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                    O.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
            }));
    // It is also possible to dispatch over all types with AT_DISPATCH_ALL_TYPES

    return {O};
}

/**
 *  - Write a c++ function that allocates tensors for the derivatives and calls
 *  the kernels to compute
 *  their content.
 */
std::vector<torch::Tensor> center_surround_convolution_backward(torch::Tensor dL_dO,
                                                                torch::Tensor I, torch::Tensor w_c,
                                                                torch::Tensor w_s,
                                                                torch::Tensor w_b)
{
    // Check inputs
    CHECK_INPUT(dL_dO);
    CHECK_INPUT(w_c);
    CHECK_INPUT(w_s);
    CHECK_INPUT(w_b);

    // Allocate memory for dL_dI, dL_dw_c, dL_dw_s and dL_dw_b
    auto dL_dI = torch::empty_like(I);
    auto dL_dw_c = torch::empty_like(w_c);
    auto dL_dw_s = torch::empty_like(w_s);
    auto dL_dw_b = torch::empty_like(w_b);

    auto C_i = w_c.size(0);
    auto C_o = w_c.size(1);
    auto H = I.size(2);
    auto W = I.size(3);

    // Call the kernels with correct grid and block sizes
    // Each derivative is calculate on it's own.

    // Compute derivatives with respect to weights
    const dim3 block_dw(32, 32);
    const dim3 grid_dw((C_i - 1) / 32 + 1, (C_o - 1) / 32 + 1);

    // This launches the kernel
    AT_DISPATCH_FLOATING_TYPES(
        I.scalar_type(), "dL_dw_kernel",
        (
            [&]
            {
                dL_dw_kernel<scalar_t><<<grid_dw, block_dw>>>(
                    dL_dO.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    I.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    dL_dw_c.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    dL_dw_s.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
            }));

    const dim3 block_db(1024);
    const dim3 grid_db((C_o - 1) / 1024 + 1);
    AT_DISPATCH_FLOATING_TYPES(
        I.scalar_type(), "dL_dw_b_kernel",
        (
            [&]
            {
                dL_dw_b_kernel<scalar_t><<<grid_db, block_db>>>(
                    dL_dO.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    dL_dw_b.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
            }));

    const dim3 block_dI(32, 32);
    const dim3 grid_dI((W - 1) / 32 + 1, (H - 1) / 32 + 1);

    // Zero padding of dL_dO allows the reuse of the forward path
    // The list contains padding sizes starting at the back of the tensor
    // {last dim before, last dim after, second to last before, second to last
    // after, ...}
    dL_dO = torch::constant_pad_nd(dL_dO,
                                   torch::IntList({2, 2, 2, 2}), // padding of 2 around Y and X dim.
                                   0);

    AT_DISPATCH_FLOATING_TYPES(
        I.scalar_type(), "dL_dI_kernel",
        (
            [&]
            {
                dL_dI_kernel<scalar_t><<<grid_dI, block_dI>>>(
                    dL_dO.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    w_c.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    w_s.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    dL_dI.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
            }));

    // Each gradient needs to be returned for the backpropagation
    return {dL_dI, dL_dw_c, dL_dw_s, dL_dw_b};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &center_surrond_convolution_forward,
          "Center-Surround-Convolution TODO documentation string");
    m.def("backward", &center_surround_convolution_backward,
          "Center-Surround-Convolution TODO documentation string");
}
