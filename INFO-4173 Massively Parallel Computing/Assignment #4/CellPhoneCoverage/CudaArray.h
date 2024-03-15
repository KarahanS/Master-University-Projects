#ifndef _CudaArray_h_
#define _CudaArray_h_

#include <cuda_runtime.h>

//////////////////////////////////////////////////////////////////////////
//
// Convenience class, which tracks an array in CUDA device memory
//
// It performs allocation/deallocation of the array,
//  and allows to copy the entire array's contents to/from device memory
//

template <typename T> class CudaArray
{
private:
    T* m_cudaArray;

    int m_size;

public:
    // Create array
    CudaArray(int size);

    // Destroy array
    ~CudaArray();

    // Retrieve pointer to array (in device memory)
    T* cudaArray() const;

    // Retrieve size of array
    int size() const;

    // Copy entire array; from device memory into user-specified array
    //  in host memory
    void copyFromCuda(T* cpuArray);

    // Copy entire array; from user-specified array in host memory
    //  to the array in device memory
    void copyToCuda(const T* cpuArray);
};

template <typename T> CudaArray<T>::CudaArray(int size)
{
    m_size = size;
    void* obj;
    cudaMalloc(&obj, sizeof(T) * m_size);
    m_cudaArray = reinterpret_cast<T*>(obj);
}

template <typename T> CudaArray<T>::~CudaArray() { cudaFree(m_cudaArray); }

template <typename T> T* CudaArray<T>::cudaArray() const { return m_cudaArray; }

template <typename T> int CudaArray<T>::size() const { return m_size; }

template <typename T> void CudaArray<T>::copyFromCuda(T* cpuArray)
{
    cudaMemcpy(cpuArray, m_cudaArray, sizeof(T) * m_size, cudaMemcpyDeviceToHost);
}

template <typename T> void CudaArray<T>::copyToCuda(const T* cpuArray)
{
    cudaMemcpy(m_cudaArray, cpuArray, sizeof(T) * m_size, cudaMemcpyHostToDevice);
}

#endif // _CudaArray_h_
