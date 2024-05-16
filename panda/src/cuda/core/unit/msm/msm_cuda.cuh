#pragma once

#include <chrono>

#include "panda_interface.cuh"
#include "common/common.cuh"
#include "msm_config.cuh"
#include "curve/curve.cuh"
#include "curve/bn254/config.cuh"

using namespace common;
using namespace panda_curve;

namespace panda_msm_v1
{
    constexpr unsigned log2(unsigned value)
    {
        return (value <= 1) ? 0 : 1 + log2(value / 2);
    }

    static unsigned get_window_bits_count(const unsigned log_scalars_count)
    {
        switch (log_scalars_count)
        {
        case 13:
        case 14:
            return 14;
        case 15:
        case 16:
        case 17:
        case 18:
        case 19:
            return 15;
        case 20:
        case 21:
        case 22:
        case 23:
        case 24:
        case 25:
        case 26:
            return 16;
        default:
            return max(log_scalars_count, 3u);
        }
    }

    template <typename Fr, typename Affine, typename Projective>
    struct execution_configuration
    {
        cudaMemPool_t mem_pool;
        cudaStream_t stream;
        Affine *bases;
        Fr *scalars;
        Projective *results;
        unsigned log_scalars_count;
        RESULT_COORDINATE_TYPE msm_result_coordinate_type;
    };

    template <typename Projective>
    static unsigned calc_groups_sums(Projective &result, Projective *groups, const unsigned groups_num, const unsigned bit_s)
    {
        Projective group_sum = {0x0};
        Projective g0 = groups[0];

        for (unsigned i = 0; i < (groups_num - 1); i++)
        {
            group_sum = group_sum + groups[groups_num - i - 1];
            for (unsigned j = 0; j < bit_s; j++)
            {
                group_sum = Projective::dbl(group_sum);
            }
        }

        result = group_sum + g0;

        return 0;
    }

    template <typename Projective>
    __global__ void calc_groups_sums_cuda(Projective *result, Projective *groups, unsigned groups_num)
    {
        Projective group_sum = {0x0};
        Projective g0 = groups[0];

        for (unsigned i = 0; i < (groups_num - 1); i++)
        {
            group_sum = group_sum + groups[groups_num - i - 1];

            for (unsigned j = 0; j < BIT_S; j++)
            {
                group_sum = Projective::dbl(group_sum);
            }
        }

        *result = group_sum + g0;
    }

    template <typename Projective>
    __global__ void calc_groups_sums_cuda(Projective *result, Projective *groups, unsigned groups_num, unsigned threads)
    {
        const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

        if (tid < threads)
        {
            Projective group_sum = {0x0};
            Projective g0 = groups[0];

            for (unsigned i = 0; i < (groups_num - 1); i++)
            {
                group_sum = group_sum + groups[groups_num - i - 1];

                for (unsigned j = 0; j < BIT_S; j++)
                {
                    group_sum = Projective::dbl(group_sum);
                }
            }

            result[tid] = group_sum + g0;
        }
    }


    template <typename Affine, typename Projective>
    __global__ void calc_groups_sums_to_affine_cuda(Affine *result, Projective *groups, unsigned groups_num, unsigned threads)
    {
        const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

        if (tid < threads)
        {
            Projective group_sum = {0x0};
            // Projective g0 = groups[0];

            for (unsigned i = 0; i < (groups_num - 1); i++)
            {
                group_sum = group_sum + groups[groups_num - i - 1];

                for (unsigned j = 0; j < BIT_S; j++)
                {
                    group_sum = Projective::dbl(group_sum);
                }
            }

            group_sum = group_sum + groups[0];
            result[tid] = Projective::to_affine(group_sum);
        }
    }

    template <typename Fr>
    __global__ void init_handle_scalars_kernel(Fr *__restrict__ d_scalars, const unsigned count)
    {
        const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

        if (tid < count)
        {
            d_scalars[tid] = Fr::from_montgomery(d_scalars[tid]);
        }
    }

    template <typename Fr>
    __global__ void calc_lens_for_per_thread_kernel(
        const Fr *__restrict__ d_scalars,
        const unsigned msm_size,
        const unsigned groups_num,
        const unsigned bit_s,
        unsigned *__restrict__ d_arr_len_per_thread,
        const unsigned block_dim_exp)
    {
        unsigned last_group_bits = Fr::BC % bit_s;
        for (unsigned id = (blockIdx.x << block_dim_exp) + threadIdx.x;
             id < msm_size; id += (gridDim.x << block_dim_exp))
        {
            const Fr local_scalar = d_scalars[id];

            for (unsigned i = 0; i < groups_num; i++)
            {
                const unsigned index_start = i * bit_s;
                const unsigned index_end = i < groups_num - 1 ? index_start + bit_s - 1 : Fr::BC - 1;

                const unsigned m = Fr::LB == 0 ? 0 : index_start / Fr::LB;
                const unsigned n = Fr::LB == 0 ? 0 : index_end / Fr::LB;
                const unsigned m_shift = index_start - Fr::LB * m;

                unsigned scalar_slice = 0;
                const unsigned current_bits = last_group_bits ? (i < groups_num - 1 ? bit_s : last_group_bits) : bit_s;

                if (m == n)
                {
                    unsigned long long slice_long = local_scalar.limbs_storage.limbs[m] << (Fr::LB - current_bits - m_shift);
                    scalar_slice = slice_long >> (Fr::LB - current_bits);
                }
                else
                {
                    scalar_slice = local_scalar.limbs_storage.limbs[m] >> m_shift;
                    unsigned long long slice_high = local_scalar.limbs_storage.limbs[n] << ((Fr::LB << 1) - current_bits - m_shift);
                    slice_high = slice_high >> (Fr::LB - current_bits);
                    scalar_slice = scalar_slice + slice_high;
                }

                if (scalar_slice)
                {
                    atomicAdd(d_arr_len_per_thread + ((i << bit_s) - i + scalar_slice - 1), 1);
                }
            }
        }
    }

    static __global__ void allo_arrs_for_per_thread_kernel(
        unsigned long long *__restrict__ d_origin_base_id_addr_per_group,
        const unsigned *__restrict__ d_arr_len_per_thread,
        const unsigned total_buckets_num,
        const unsigned bit_s,
        unsigned **__restrict__ d_arr_address_per_thread,
        unsigned *__restrict__ d_arr_len_per_thread1,
        const unsigned block_dim_exp)
    {
        for (unsigned id = (blockIdx.x << block_dim_exp) + threadIdx.x;
             id < total_buckets_num; id += (gridDim.x << block_dim_exp))
        {
            unsigned len = d_arr_len_per_thread[id];

            if (len)
            {
                unsigned *ptr = (unsigned *)atomicAdd(d_origin_base_id_addr_per_group + id / ((1 << bit_s) - 1), len * sizeof(unsigned));
                d_arr_address_per_thread[id] = ptr;
                // memset(ptr, 0, len * sizeof(unsigned));
            }

            d_arr_len_per_thread1[id] = len;
        }
    }

    template <typename Fr>
    __global__ void fill_arrs_for_per_thread_kernel(
        const Fr *__restrict__ d_scalars,
        unsigned *__restrict__ d_arr_len_per_thread,
        unsigned **__restrict__ d_arr_address_per_thread,
        const unsigned groups_num,
        const unsigned bit_s,
        const unsigned count,
        const unsigned block_dim_exp)
    {
        auto last_bits = Fr::BC % bit_s;

        for (unsigned id = (blockIdx.x << block_dim_exp) + threadIdx.x;
                id < count; id += (gridDim.x << block_dim_exp))
        {
            const Fr local_scalar = d_scalars[id];

            for (unsigned i = 0; i < groups_num; ++i)
            {
                const unsigned index_start = i * bit_s;
                const unsigned index_end = i < groups_num - 1 ? index_start + bit_s - 1 : Fr::BC - 1;

                const unsigned m = Fr::LB == 0 ? 0 : index_start / Fr::LB;
                const unsigned n = Fr::LB == 0 ? 0 : index_end / Fr::LB;
                const unsigned m_shift = index_start - Fr::LB * m;

                unsigned scalar_slice = 0;
                const unsigned current_bits = last_bits ? (i < groups_num - 1 ? bit_s : last_bits): bit_s;

                if (m == n)
                {
                    unsigned long long slice_long = local_scalar.limbs_storage.limbs[m] << (Fr::LB - current_bits - m_shift);
                    scalar_slice = slice_long >> (Fr::LB - current_bits);
                }
                else
                {
                    scalar_slice = local_scalar.limbs_storage.limbs[m] >> m_shift;
                    unsigned long long slice_high = local_scalar.limbs_storage.limbs[n] << ((Fr::LB << 1) - current_bits - m_shift);
                    slice_high = slice_high >> (Fr::LB - current_bits);
                    scalar_slice = scalar_slice + slice_high;
                }

                if (scalar_slice)
                {
                    const unsigned total_buckets_id = (i << bit_s) - i + scalar_slice - 1;
                    d_arr_address_per_thread[total_buckets_id]
                                            [atomicSub(d_arr_len_per_thread + total_buckets_id, 1) - 1] = id;
                }
            }
        }
    }

    template <typename Projective>
    __device__ void reduceSumInWarp(Projective &tmp, Projective &buktResult, const unsigned &mask, unsigned lanes)
    {
        while (lanes > 1)
        {
            for (int i = 0; i < tmp.x.LC; i++)
            {
                tmp.x.limbs_storage.limbs[i] = __shfl_down_sync(mask, buktResult.x.limbs_storage.limbs[i], lanes + (lanes & 0x1) >> 1);
                tmp.y.limbs_storage.limbs[i] = __shfl_down_sync(mask, buktResult.y.limbs_storage.limbs[i], lanes + (lanes & 0x1) >> 1);
                tmp.z.limbs_storage.limbs[i] = __shfl_down_sync(mask, buktResult.z.limbs_storage.limbs[i], lanes + (lanes & 0x1) >> 1);
            }

            if ((threadIdx.x & 0x1f) < lanes >> 1)
                buktResult = buktResult + tmp;

            lanes = lanes + (lanes & 0x1) >> 1;
        }
    }

#define MAX_THREADS 1024
    template <typename Fr>
    __host__ cudaError_t init_handle_scalars(Fr *d_scalars, const unsigned count, cudaStream_t stream)
    {
        const dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
        const dim3 grid_dim = (count - 1) / block_dim.x + 1;

        init_handle_scalars_kernel<<<grid_dim, block_dim, 0, stream>>>(d_scalars, count);

        return cudaGetLastError();
    }
#undef MAX_THREADS

#define MAX_THREADS 256
    template <typename Fr>
    __host__ cudaError_t calc_lens_for_per_thread(
        Fr *d_scalars,
        const unsigned count,
        const unsigned groups_num,
        const unsigned bit_s,
        unsigned *__restrict__ d_arr_len_per_thread,
        cudaStream_t stream)
    {
        const dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
        const dim3 grid_dim = (count - 1) / block_dim.x + 1;
        unsigned block_dim_exp = log2(MAX_THREADS);
        calc_lens_for_per_thread_kernel<<<grid_dim, block_dim, 0, stream>>>(d_scalars, count, groups_num, bit_s,
                                                                            d_arr_len_per_thread, block_dim_exp);

        return cudaGetLastError();
    }

    __host__ cudaError_t allo_arrs_for_per_thread(
        unsigned long long *__restrict__ d_origin_base_id_addr_per_group,
        unsigned *__restrict__ d_arr_len_per_thread,
        unsigned **__restrict__ d_arr_address_per_thread,
        unsigned *__restrict__ d_arr_len_per_thread1,
        const unsigned total_buckets_num,
        const unsigned bit_s,
        cudaStream_t stream)
    {
        const dim3 block_dim = total_buckets_num < MAX_THREADS ? total_buckets_num : MAX_THREADS;
        const dim3 grid_dim = (total_buckets_num - 1) / block_dim.x + 1;
        unsigned block_dim_exp = log2(MAX_THREADS);
        allo_arrs_for_per_thread_kernel<<<grid_dim, block_dim, 0, stream>>>(d_origin_base_id_addr_per_group, d_arr_len_per_thread, total_buckets_num, bit_s,
                                                                            d_arr_address_per_thread, d_arr_len_per_thread1, block_dim_exp);

        return cudaGetLastError();
    }

    template <typename Fr>
    __host__ cudaError_t fill_arrs_for_per_thread(
        Fr *d_scalars,
        unsigned *__restrict__ d_arr_len_per_thread,
        unsigned **__restrict__ d_arr_address_per_thread,
        const unsigned count,
        const unsigned groups_num,
        const unsigned bit_s,
        cudaStream_t stream)
    {
        const dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
        const dim3 grid_dim = (count - 1) / block_dim.x + 1;
        unsigned block_dim_exp = log2(MAX_THREADS);
        fill_arrs_for_per_thread_kernel<<<grid_dim, block_dim, 0, stream>>>(d_scalars, d_arr_len_per_thread, d_arr_address_per_thread,
                                                                            groups_num, bit_s, count, block_dim_exp);

        return cudaGetLastError();
    }
#undef MAX_THREADS

    template <typename Affine, typename Projective>
    __global__ void aggerate_buckets_groups_kernel(
        const Affine *__restrict__ d_bases,
        const unsigned totalThreads,
        unsigned **__restrict__ d_arr_address_per_thread,
        const unsigned *__restrict__ d_arr_len_per_thread1,
        Projective *__restrict__ d_groups,
        const unsigned bit_s,
        const unsigned groups_num,
        const unsigned last_group_bits,
        const unsigned agg_block_max_dim_exp,
        const unsigned blocks_per_group,
        const unsigned agg_block_dim,
        const unsigned agg_block_dim_exp)
    {
        extern __shared__ unsigned char shared_memory[];
        Projective *sharedData = reinterpret_cast<Projective *>(shared_memory);

        unsigned looped_times = 0;
        for (unsigned id = (blockIdx.x << agg_block_dim_exp) + threadIdx.x;
             id < totalThreads; id += (gridDim.x << agg_block_dim_exp))
        {
            Projective buktResult = {0x0};
            unsigned blocksBefore = blockIdx.x + looped_times * gridDim.x;
            unsigned value = (blocks_per_group > 1 ? ((blocksBefore % blocks_per_group) == (blocks_per_group - 1)
                                                          ? agg_block_dim - 1
                                                          : agg_block_dim)
                                                   : agg_block_dim - 1);
            const unsigned currentBlockThreads = last_group_bits ? (blocksBefore + 1 < gridDim.x ? value : (last_group_bits > agg_block_max_dim_exp ? agg_block_dim - 1 : (1 << last_group_bits) - 1)) : value;
            if (threadIdx.x < currentBlockThreads)
            {
                unsigned len = d_arr_len_per_thread1[id - blocksBefore / blocks_per_group];
                Projective buktInit = {0x0};
                for (unsigned i = 0; i < len; ++i)
                {
                    buktInit = buktInit + d_bases[d_arr_address_per_thread[id - blocksBefore / blocks_per_group][i]];
                }

                unsigned buktsPrefix = (blocksBefore % blocks_per_group << agg_block_dim_exp) + threadIdx.x + 1;

                for (int i = bit_s - 1; i >= 0; --i)
                {
                    buktResult = Projective::dbl(buktResult);
                    if ((buktsPrefix >> i) & 0x01)
                    {
                        buktResult = buktResult + buktInit;
                    }
                }

                unsigned mask = __activemask();
                unsigned lanes = __popc(mask);
                reduceSumInWarp(buktInit, buktResult, mask, lanes);

                if (currentBlockThreads > 32)
                {
                    if (!(threadIdx.x & 0x1f))
                        sharedData[threadIdx.x >> 5] = buktResult;
                    __syncthreads();

                    if (threadIdx.x < (1 << agg_block_dim_exp - 5))
                    {
                        buktResult = sharedData[threadIdx.x];

                        mask = __activemask();
                        lanes = __popc(mask);
                        reduceSumInWarp(buktInit, buktResult, mask, lanes);
                    }
                }

                if (!threadIdx.x)
                {
                    d_groups[blocksBefore] = buktResult;
                }
            }
            looped_times++;
        }
    }

    template <typename Projective>
    __global__ void groups_sum_blocks_kernel(
        Projective *__restrict__ d_final_groups,
        const Projective *__restrict__ d_groups,
        const unsigned totalBlocksInPre,
        const unsigned bit_s,
        const unsigned last_group_bits,
        unsigned agg_block_max_dim_exp,
        unsigned blocks_per_group)
    {
        extern __shared__ Projective sharedData[];

        if ((blockIdx.x << bit_s - agg_block_max_dim_exp) + threadIdx.x < totalBlocksInPre)
        {
            Projective my = d_groups[(blockIdx.x << bit_s - agg_block_max_dim_exp) + threadIdx.x];

            Projective tmp = {0x0};

            unsigned mask = __activemask();
            unsigned lanes = __popc(mask);
            reduceSumInWarp(tmp, my, mask, lanes);

            const unsigned threadsThisBlock = blockIdx.x < gridDim.x - 1 ? blocks_per_group
                                                                         : (last_group_bits > agg_block_max_dim_exp ? 1 << last_group_bits - agg_block_max_dim_exp
                                                                                                                    : (last_group_bits ? 1 : blocks_per_group));
            if (threadsThisBlock > 32)
            {
                if (!(threadIdx.x & 0x1f))
                    sharedData[threadIdx.x >> 5] = my;
                __syncthreads();

                if (threadIdx.x < (threadsThisBlock >> 5))
                {
                    my = sharedData[threadIdx.x];

                    mask = __activemask();
                    lanes = __popc(mask);
                    reduceSumInWarp(tmp, my, mask, lanes);
                }
            }

            if (!threadIdx.x)
            {
                d_final_groups[blockIdx.x] = my;
            }
        }
    }

#define MAX_THREADS 256
    template <typename Affine, typename Projective>
    __host__ cudaError_t aggerate_buckets_groups(
        const Affine *__restrict__ d_bases,
        const unsigned bit_s,
        const unsigned count,
        const unsigned last_group_bits,
        unsigned **__restrict__ d_arr_address_per_thread,
        const unsigned *__restrict__ d_arr_len_per_thread1,
        Projective *__restrict__ d_groups,
        const unsigned groups_num,
        cudaStream_t stream)
    {
        unsigned agg_block_max_dim_exp = log2(MAX_THREADS);
        unsigned blocks_per_group = bit_s > agg_block_max_dim_exp ? 1 << (bit_s - agg_block_max_dim_exp) : 1;
        unsigned agg_block_dim = (1 << bit_s) > MAX_THREADS ? MAX_THREADS : (1 << bit_s);
        unsigned agg_block_dim_exp = log2(agg_block_dim);

        const dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
        const dim3 grid_dim = (count - 1) / block_dim.x + 1;

        aggerate_buckets_groups_kernel<<<grid_dim, block_dim, 0, stream>>>(d_bases, count, d_arr_address_per_thread,
                                                                           d_arr_len_per_thread1, d_groups, bit_s, groups_num, last_group_bits, agg_block_max_dim_exp, blocks_per_group, agg_block_dim, agg_block_dim_exp);

        return cudaGetLastError();
    }

    template <typename Projective>
    __host__ cudaError_t groups_sum_blocks(
        Projective *__restrict__ d_final_groups,
        Projective *__restrict__ d_groups,
        const unsigned bit_s,
        const unsigned last_group_bits,
        const unsigned count,
        const unsigned groups_num,
        cudaStream_t stream)
    {
        unsigned agg_block_max_dim_exp = log2(MAX_THREADS);
        unsigned blocks_per_group = bit_s > agg_block_max_dim_exp ? 1 << (bit_s - agg_block_max_dim_exp) : 1;
        int shiftBits = bit_s - static_cast<int>(agg_block_max_dim_exp) - 5;
        unsigned sharedSize = (shiftBits > 0 ? 1 << shiftBits : 0) * sizeof(Projective);

        const dim3 block_dim = MAX_THREADS;
        const dim3 grid_dim = (count - 1) / block_dim.x + 1;

        groups_sum_blocks_kernel<<<groups_num, blocks_per_group, sharedSize, stream>>>(d_final_groups, d_groups, grid_dim.x, bit_s, last_group_bits,
                                                                                       agg_block_max_dim_exp, blocks_per_group);

        return cudaGetLastError();
    }
#undef MAX_THREADS

    template <typename Fr, typename Affine, typename Projective>
    cudaError_t msm_execute_cuda(const execution_configuration<Fr, Affine, Projective> &exec_cfg)
    {
        unsigned deviceID = 0;
        cudaSetDevice(deviceID);

        cudaStream_t stream = exec_cfg.stream;
        cudaMemPool_t mem_pool = exec_cfg.mem_pool;
        Affine *d_bases = exec_cfg.bases;
        Fr *d_scalars = exec_cfg.scalars;
        Projective *results = exec_cfg.results;
        unsigned log_scalars_count = exec_cfg.log_scalars_count;
        unsigned msm_size = 1 << log_scalars_count;
        unsigned bit_s = get_window_bits_count(log_scalars_count);
        unsigned groups_num = (Fr::BC + (bit_s - 1)) / bit_s;
        unsigned total_buckets_num = (Fr::BC / bit_s << bit_s) - Fr::BC / bit_s + (1 << Fr::BC % bit_s) - 1;

        Projective *h_groups = new Projective[groups_num];
        Projective *d_groups = nullptr;
        Projective *d_final_groups = nullptr;
        unsigned *d_arr_len_per_thread = nullptr;
        unsigned *d_arr_len_per_thread1 = nullptr;
        unsigned **d_arr_address_per_thread = nullptr;
        unsigned *d_buckets_base_index = nullptr;
        unsigned long long *d_origin_base_id_addr_per_group = nullptr;

#if DEBUG_LOG
        Affine *bases = new Affine[msm_size];
        cudaMemcpyAsync(bases, d_bases, msm_size * sizeof(Affine), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        Fr *scalars = new Fr[msm_size];
        cudaMemcpyAsync(scalars, d_scalars, msm_size * sizeof(Fr), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        printf("    log_scalars_count: %d, msm_size: %d, bit_s: %d, groups_num: %d\n", log_scalars_count, msm_size, bit_s, groups_num);
        Affine::print(bases[0]);
        Affine::print(Affine::from_montgomery(bases[0]));
        // Affine::print(bases[1]);
        Affine::print(bases[msm_size - 1]);
        Affine::print(Affine::from_montgomery(bases[msm_size - 1]));
    
        Fr::print(scalars[0]);
        Fr::print(Fr::from_montgomery(scalars[0]));
        // Fr::print(scalars[1]);
        Fr::print(scalars[msm_size - 1]);
        Fr::print(Fr::from_montgomery(scalars[msm_size - 1]));

        delete[] bases;
        delete[] scalars;
#endif

        // allocate memory
        HANDLE_RESULT_CUDA(cudaMallocAsync(&d_groups, groups_num * sizeof(Projective), stream));
        HANDLE_RESULT_CUDA(cudaMallocAsync(&d_final_groups, groups_num * sizeof(Projective), stream));
        HANDLE_RESULT_CUDA(cudaMallocAsync(&d_arr_len_per_thread, total_buckets_num * sizeof(unsigned), stream));
        HANDLE_RESULT_CUDA(cudaMallocAsync(&d_arr_len_per_thread1, total_buckets_num * sizeof(unsigned), stream));
        HANDLE_RESULT_CUDA(cudaMallocAsync(&d_arr_address_per_thread, total_buckets_num * sizeof(unsigned *), stream));
        HANDLE_RESULT_CUDA(cudaMallocAsync(&d_buckets_base_index, groups_num * msm_size * sizeof(unsigned *), stream));
        HANDLE_RESULT_CUDA(cudaMallocAsync(&d_origin_base_id_addr_per_group, groups_num * sizeof(unsigned long long), stream));
        cudaDeviceSynchronize();

        unsigned long long h_origin_base_id_addr_per_group[groups_num];
        for (unsigned i = 0; i < groups_num; i++)
        {
            h_origin_base_id_addr_per_group[i] = (unsigned long long)(d_buckets_base_index + (i << log_scalars_count));
        }

        HANDLE_RESULT_CUDA(cudaMemcpyAsync(d_origin_base_id_addr_per_group, h_origin_base_id_addr_per_group, groups_num * sizeof(unsigned long long),
                                           cudaMemcpyHostToDevice, stream));

        cudaDeviceSynchronize();

#if DEBUG_PROFILING
        auto start1 = chrono::high_resolution_clock::now();
        uint64_t t0 = __builtin_ia32_rdtsc();
        cudaEvent_t start, stop;
        float cudaTime = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
#endif

        // 1
        HANDLE_RESULT_CUDA(init_handle_scalars(d_scalars, msm_size, stream));
        cudaDeviceSynchronize();

#if DEBUG_PROFILING
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cudaTime, start, stop);
        printf("-----   init_handle_scalars time: %f ms\n", cudaTime);
        cudaEventRecord(start, stream);
#endif

        // 2
        HANDLE_RESULT_CUDA(calc_lens_for_per_thread(d_scalars, msm_size, groups_num, bit_s, d_arr_len_per_thread, stream));
        cudaDeviceSynchronize();

#if DEBUG_PROFILING
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cudaTime, start, stop);
        printf("-----   calc_lens_for_per_thread time: %f ms\n", cudaTime);
        cudaEventRecord(start, stream);
#endif

        // 3
        HANDLE_RESULT_CUDA(allo_arrs_for_per_thread(
            d_origin_base_id_addr_per_group,
            d_arr_len_per_thread,
            d_arr_address_per_thread,
            d_arr_len_per_thread1,
            total_buckets_num,
            bit_s,
            stream));

        cudaDeviceSynchronize();
#if DEBUG_PROFILING
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cudaTime, start, stop);
        printf("-----   allo_arrs_for_per_thread time: %f ms\n", cudaTime);
        cudaEventRecord(start, stream);
#endif

        // 4
        HANDLE_RESULT_CUDA(fill_arrs_for_per_thread(
            d_scalars,
            d_arr_len_per_thread,
            d_arr_address_per_thread,
            msm_size,
            groups_num,
            bit_s,
            stream));

        cudaDeviceSynchronize();
#if DEBUG_PROFILING
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cudaTime, start, stop);
        printf("-----   fill_arrs_for_per_thread time: %f ms\n", cudaTime);
        cudaEventRecord(start, stream);
#endif

        // 5
        unsigned count = (Fr::BC / bit_s << bit_s) + (1 << Fr::BC % bit_s) - 1;
        unsigned last_group_bits = Fr::BC % bit_s;
        HANDLE_RESULT_CUDA(aggerate_buckets_groups(
            d_bases,
            bit_s,
            count,
            last_group_bits,
            d_arr_address_per_thread,
            d_arr_len_per_thread1,
            d_groups,
            groups_num,
            stream));

        cudaDeviceSynchronize();
#if DEBUG_PROFILING
        cudaStreamSynchronize(stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cudaTime, start, stop);
        printf("-----   aggerate_buckets_groups time: %f ms\n", cudaTime);
        cudaEventRecord(start, stream);
#endif

        // 6
        HANDLE_RESULT_CUDA(groups_sum_blocks(
            d_final_groups,
            d_groups,
            bit_s,
            last_group_bits,
            count,
            groups_num,
            stream));

        cudaDeviceSynchronize();
#if DEBUG_PROFILING
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cudaTime, start, stop);
        printf("-----   groups_sum_blocks time: %f ms\n", cudaTime);
#endif

        cudaMemcpyAsync(h_groups, d_final_groups, groups_num * sizeof(Projective), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // 7
        Projective *h_results = new Projective[1];
        calc_groups_sums(*h_results, h_groups, groups_num, bit_s);

        if (exec_cfg.msm_result_coordinate_type == RESULT_COORDINATE_TYPE::RESULT_COORDINATE_TYPE_PROJECTIVE)
        {
            *h_results = Projective::to_projective(*h_results);
        }

        // print result
        // Projective::print(*h_results);
        // Projective::print(Projective::from_montgomery(*h_results));

        cudaMemcpyAsync(results, h_results, sizeof(Projective), cudaMemcpyHostToDevice, stream);
        cudaDeviceSynchronize();

        cudaFreeAsync(d_origin_base_id_addr_per_group, stream);
        cudaFreeAsync(d_buckets_base_index, stream);
        cudaFreeAsync(d_groups, stream);
        cudaFreeAsync(d_final_groups, stream);
        cudaFreeAsync(d_arr_len_per_thread, stream);
        cudaFreeAsync(d_arr_len_per_thread1, stream);
        cudaFreeAsync(d_arr_address_per_thread, stream);

        delete[] h_groups;
        delete[] h_results;

        return cudaSuccess;
    }

    cudaError_t core_msm_execute_bn254(const msm_configuration configuration)
    {
        execution_configuration<CURVE_BN254::fr, CURVE_BN254::affine, CURVE_BN254::projective> exe_cfg = {
            static_cast<cudaMemPool_t>(configuration.mem_pool.handle),
            static_cast<cudaStream_t>(configuration.stream.handle),
            static_cast<CURVE_BN254::affine *>(configuration.bases),
            static_cast<CURVE_BN254::fr *>(configuration.scalars),
            static_cast<CURVE_BN254::projective *>(configuration.results),
            configuration.log_scalars_count,
            static_cast<RESULT_COORDINATE_TYPE>(configuration.msm_result_coordinate_type),
        };

        return msm_execute_cuda<CURVE_BN254::fr, CURVE_BN254::affine, CURVE_BN254::projective>(exe_cfg);
    }

    template <typename S>
    static cudaError_t set_up()
    {
        return cudaSuccess;
    }

    cudaError_t core_msm_setup_bn254()
    {
        return set_up<CURVE_BN254::fr>();
    }

    static cudaError_t tear_down()
    {
        return cudaSuccess;
    }

    cudaError_t core_msm_tear_down()
    {
        return tear_down();
    }
}
