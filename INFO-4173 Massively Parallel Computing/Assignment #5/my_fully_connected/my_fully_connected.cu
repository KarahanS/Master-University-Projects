#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
    CHECK_CUDA(x);                                                                                 \
    CHECK_CONTIGUOUS(x)

// forward kernel
template <typename scalar_t>
__global__ void my_fully_connected_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input, // [N, I]
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> weight, // [I, O]
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> bias, // [O]
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output) // [N, O]
{

    const int o = blockIdx.x * blockDim.x + threadIdx.x; // Calculate this threads output index
    const int n = blockIdx.y * blockDim.y + threadIdx.y; // Calculate this threads batch index

    const int N = input.size(0); // Obtain batch size
    const int O = output.size(1); // Obtain output size
    const int I = input.size(1); // Obtain input size

    if (o >= O || n >= N) // Bounds checking
        return;

    // Compute result for this thread's element [n][o]
    scalar_t result = 0.0f;
    for (int i = 0; i < I; ++i)
        result += input[n][i] * weight[i][o];
    result += bias[o];

    output[n][o] = result; // Store the computed result to the thread's output element
}

// backward kernel towards input
template <typename scalar_t>
__global__ void dL_dinput_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dL_doutput, // [N, O]
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> weight, // [I, O]
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dL_dinput) // [N, I]
{

    const int i = blockIdx.x * blockDim.x + threadIdx.x; // Calculate this thread's input index
    const int n = blockIdx.y * blockDim.y + threadIdx.y; // Calculate this thread's batch index

    const int I = dL_dinput.size(1); // Obtain input size
    const int O = dL_doutput.size(1); // Obtain output size
    const int N = dL_doutput.size(0); // Obtain batch size

    if (i >= I || n >= N) // Bounds checking
        return;

    // Compute the derivative for this thread's element [n][i]
    float result = 0.0f;
    for (int o = 0; o < O; o++)
        result += weight[i][o] * dL_doutput[n][o];

    dL_dinput[n][i] = result; // Store the thread's result in the input derivative tensor
}

// backward kernel towards weight
template <typename scalar_t>

__global__ void dL_dweight_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dL_doutput, // [N, O]
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input, // [N, I]
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dL_dweight) // [I, O]
{

    const int o = blockIdx.x * blockDim.x + threadIdx.x; // Calculate this thread's output index
    const int i = blockIdx.y * blockDim.y + threadIdx.y; // Calculate this thread's input index

    const int I = input.size(1); // Obtain input size
    const int O = dL_doutput.size(1); // Obtain output size
    const int N = input.size(0); // Obtain batch size

    if (o >= O || i >= I) // Bounds checking
        return;

    // Compute the derivative for this thread's element [i][o]
    float result = 0.0f;
    for (int n = 0; n < N; n++)
        result += input[n][i] * dL_doutput[n][o];

    dL_dweight[i][o] = result; // Store the thread's result in the weight derivative tensor
}

// backward kernel towards bias
template <typename scalar_t>

__global__ void dL_dbias_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dL_doutput, // [N, O]
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dL_dbias) // [O]
{

    const int o = blockIdx.x * blockDim.x + threadIdx.x; // Calculate this thread's output index

    int N = dL_doutput.size(0); // Obtain the batch size
    int O = dL_doutput.size(1); // Obtain the output size

    if (o >= O) // Bounds checking
        return;

    // Compute the derivative for this thread's element [o]
    float result = 0.0f;
    for (int n = 0; n < N; n++)
        result += dL_doutput[n][o];

    dL_dbias[o] = result; // Store the thread's result in the weight derivative tensor
}

// forward kernel dispatching function
std::vector<torch::Tensor> my_fully_connected_forward(torch::Tensor input, // [N, I]
                                                      torch::Tensor weight, // [I, O]
                                                      torch::Tensor bias) // [O]
{
    CHECK_INPUT(input); // Checks the tensors for .is_cuda() and .is_contignuous()
    CHECK_INPUT(weight); // We provide a simple GPU implementation only
    CHECK_INPUT(bias);

    const auto N = input.size(0); // Obtain the batch size
    const auto I = input.size(1); // Obtain the input size
    const auto O = weight.size(1); // Obtain the output size

    auto output = torch::empty({N, O}, input.options()); // Create an uninitialized output tensor

    const dim3 block_dim(32, 32); // Use 1024 element blocks
    const dim3 grid_dim((O + 31) / 32,
                        (N + 31) / 32); // Map output elements to x and batch elements to y
                                        // -> One thread per calculated output element
    AT_DISPATCH_FLOATING_TYPES( // Executes the kernel only if input.type() is a floating point type
        input.type(), "my_fully_connected_forward",
        ([&] { // and sets the kernel's template parameter accordingly.
            my_fully_connected_forward_kernel<scalar_t><<<grid_dim, block_dim>>>(
                input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                weight.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
        }));
    return {output};
}

// backward kernel dispatching function
std::vector<torch::Tensor> my_fully_connected_backward(torch::Tensor dL_doutput, // [N, O]
                                                       torch::Tensor input, // [N, I]
                                                       torch::Tensor weight, // [I, O]
                                                       torch::Tensor bias) // [O]
{
    CHECK_INPUT(dL_doutput);
    CHECK_INPUT(input); // Checks the tensors for .is_cuda() and .is_contignuous()
    CHECK_INPUT(weight); // We provide a simple GPU implementation only
    CHECK_INPUT(bias);

    auto dL_dinput =
        torch::empty_like(input); // Create uninitialized tensors for the computed derivatives
    auto dL_dweight = torch::empty_like(weight);
    auto dL_dbias = torch::empty_like(bias);

    auto N = input.size(0); // Obtain the batch size
    auto I = weight.size(0); // Obtain the input size
    auto O = weight.size(1); // Obtain the output size

    const dim3 block_dinput(32, 32); // Use 1024 element blocks
    const dim3 grid_dinput((I + 31) / 32,
                           (N + 31) / 32); // Map input elements to x and batch elements to y
                                           // -> One thread per calculated input derivative element
    // Dispatch the kernel for calculating dL_dinput
    AT_DISPATCH_FLOATING_TYPES(
        input.type(), "dL_dinput_kernel",
        (
            [&]
            {
                dL_dinput_kernel<scalar_t><<<grid_dinput, block_dinput>>>(
                    dL_doutput.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    weight.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    dL_dinput.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
            }));

    const dim3 block_dweight(32, 32); // Use 1024 element blocks
    const dim3 grid_dweight(
        (O + 31) / 32, (I + 31) / 32); // Map output elements to x and input elements to y
                                       // -> One thread per calculated weight derivative element
    // Dispatch the kernel for calculating dL_dinput
    AT_DISPATCH_FLOATING_TYPES(
        input.type(), "dL_dweight_kernel",
        (
            [&]
            {
                dL_dweight_kernel<scalar_t><<<grid_dweight, block_dweight>>>(
                    dL_doutput.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    dL_dweight.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
            }));

    const dim3 block_dbias(512); // Use 1024 element blocks
    const dim3 grid_dbias((O + 511) / 512); // Map output ents to x
                                            // -> One thread per calculated bias derivative

    // Dispatch the kernel for calculating dL_dbias
    AT_DISPATCH_FLOATING_TYPES(
        input.type(), "dL_dweight_kernel",
        (
            [&]
            {
                dL_dbias_kernel<scalar_t><<<grid_dbias, block_dbias>>>(
                    dL_doutput.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    dL_dbias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
            }));

    return {dL_dinput, dL_dweight, dL_dbias}; // Return all calculated derivatives
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &my_fully_connected_forward, "My Fully Connected Forward");
    m.def("backward", &my_fully_connected_backward, "My Fully Connected Forward");
}
