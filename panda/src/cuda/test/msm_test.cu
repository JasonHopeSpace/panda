
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

#include "../core/field/field.cuh"
#include "../core/curve/bn254/config.cuh"
#include "../core/curve/curve.cuh"

#define DEBUG_HOST 0
#if DEBUG_HOST
#include "../core/unit/msm/msm_host.cuh"
#else
#include "../core/panda_interface.cuh"
#endif

#define GOLDEN_BASES_PATH "../test/data/msm/k13/bases.bin"
#define GOLDEN_SCALARS_PATH "../test/data/msm/k13/scalars.bin"
using namespace common;

using namespace CURVE_BN254;

int main(int argc, char **argv)
{
    int device_id = 0;
    cudaDeviceProp props{};
    HANDLE_RESULT_CUDA(cudaGetDeviceProperties(&props, device_id));

    // init
    unsigned msm_k = 13;
    unsigned msm_size = 1 << msm_k;

    cudaStream_t* stream = new cudaStream_t[1];
    cudaError_t cudaStatus = cudaSuccess;
    for(int i = 0; i < 1; ++i)
    {
        cudaStatus = cudaStreamCreateWithFlags(stream + i, cudaStreamNonBlocking);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaStreamCreateWithFlags failed: %s   with stream id %u\n", cudaGetErrorString(cudaStatus), i);
        }
    }

    // load_data
    affine *h_bases;
    fr *h_scalars;
    projective *h_result;

    cudaMallocHost(&h_bases, msm_size * sizeof(affine));
    cudaMallocHost(&h_scalars, msm_size * sizeof(fr));
    cudaMallocHost(&h_result, sizeof(projective));

    std::ifstream file;
    file.open(GOLDEN_BASES_PATH, std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
        printf("Error: open %s failed!\n", GOLDEN_BASES_PATH);
    }

    file.read((char*)h_bases, msm_size * sizeof(affine));
    file.close();

    file.open(GOLDEN_SCALARS_PATH, std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
        printf("Error: open %s failed!\n", GOLDEN_SCALARS_PATH);
    }
    file.read((char*)h_scalars, msm_size * sizeof(fr));

    file.close();

    std::string command1, command2;
    if(argc > 1)
    {
        command1 = argv[1];
        if(argc > 2)
        command2 = argv[2];
    }

#if DEBUG_HOST
    if (command1 == "cpu")
    {
        panda_msm::execution_configuration<fr, affine, projective> cfg = {
            stream[0],
            h_bases,
            h_scalars,
            h_result,
            msm_k};

        projective * lookup_groups = nullptr, * groups = nullptr;
        panda_msm::msm_execute_async_host<fr, affine, projective>(cfg, &groups); 

        delete[] lookup_groups;
        delete[] groups;
    }
#else
    if (command1 == "gpu")
    {

        affine *d_bases = nullptr;
        HANDLE_RESULT_CUDA(cudaMallocAsync(&d_bases, sizeof(affine) * msm_size, stream[0]));
        HANDLE_RESULT_CUDA(cudaMemcpyAsync(d_bases, h_bases, sizeof(affine) * msm_size, cudaMemcpyHostToDevice, stream[0]));
        cudaStreamSynchronize(stream[0]);

        fr *d_scalars = nullptr;
        HANDLE_RESULT_CUDA(cudaMallocAsync(&d_scalars, sizeof(fr) * msm_size, stream[0]));
        HANDLE_RESULT_CUDA(cudaMemcpyAsync(d_scalars, h_scalars, sizeof(fr) * msm_size, cudaMemcpyHostToDevice, stream[0]));
        cudaStreamSynchronize(stream[0]);

        panda_mem_pool mem_pool;
        panda_mem_pool_create(&mem_pool, device_id);

        panda_stream stream;
        panda_stream_create(&stream, true);

        panda_msm_result_coordinate_type result_type = JACOBIAN;
        // panda_msm_result_coordinate_type result_type = PROJECTIVE;
        panda_msm_configuration cfg1 = {
            mem_pool,
            stream, 
            d_bases,
            d_scalars,
            h_result,
            msm_k,
            result_type};

        panda_msm_execute_bn254(cfg1); 
    }
#endif

    printf("result to_affine:\n");
    affine h_result_affine = h_result->to_affine(*h_result);
    affine::print(h_result_affine);

    return 0;
}

