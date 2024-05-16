#pragma once
#include "curve/bn254/config.cuh"
#include "panda_interface.cuh"


namespace panda_ntt
{

    template <class Field> 
    cudaError_t set_up(Field* omega, Field** omegas);

    template <class Field>
    struct execution_configuration
    {
        cudaMemPool_t mem_pool;
        cudaStream_t stream;
        Field *d_src;
        Field *d_dst;
        Field  *omegas;
        unsigned log_n;
        unsigned *flag;
    };

    cudaError_t tear_down();

    template <class Field> 
    cudaError_t execute(execution_configuration<Field> &exec_cfg);

    cudaError_t core_ntt_setup_bn254(void *input);
    cudaError_t core_ntt_execute_bn254(ntt_configuration configuration);
    cudaError_t core_ntt_tear_down();

    cudaError_t core_ntt_execute_bn254_v1(panda_ntt_configuration_v1 configuration);

}