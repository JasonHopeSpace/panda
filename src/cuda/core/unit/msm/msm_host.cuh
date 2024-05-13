#pragma once

#include <iomanip>
#include "panda_interface.cuh"
#include "common/common.cuh"
#include "common/math/math.cuh"
#include "config.cuh"
#include "msm_config.cuh"

using namespace common;

namespace panda_msm
{
    constexpr unsigned lookupBits = BIT_S /*  + 2 */;

    template <typename Fr, typename Affine, typename Projective>
    struct execution_configuration2
    {
        cudaMemPool_t mem_pool;
        cudaStream_t stream;
        Affine *bases;
        Fr *scalars;
        Projective *results;
        unsigned log_scalars_count;
    };

    template <typename Fr, typename Affine, typename Projective>
    struct execution_configuration
    {
        cudaStream_t stream;
        Affine *bases;
        Fr *scalars;
        Projective *results;
        unsigned log_scalars_count;
    };

    template <typename Fr>
    static unsigned get_group_number(unsigned bit_s)
    {
        return (Fr::BC + (bit_s - 1)) / bit_s;
    }

    static unsigned get_bit_s()
    {
        return BIT_S;
    }

    #define BIT_S_MASK (uint32_t)(pow(2, BIT_S) - 1) // 7bit ->  0b1111111 = 0x7F

    template <typename Fr>
    static unsigned get_slice(Fr &scalar, unsigned index_start, unsigned index_end, unsigned *slice, unsigned slice_bit)
    {
        // host
        /* Calcuate the common position in the scalar array*/
        unsigned m = (Fr::LB == 0) ? 0 : (index_start / (Fr::LB));
        unsigned n = (Fr::LB == 0) ? 0 : (index_end / (Fr::LB));
        unsigned m_shift = index_start - Fr::LB * m;
        unsigned n_shift = index_end - Fr::LB * n;

        unsigned scalar_cut = 0;
        uint128_t scalar_joint = 0;

        /* Loop to calculate scalar_cut(each value of the bucket_index_area) of each scalar */
        if (m == n)
        {
            /* Case: m and n are in the same scalar area index*/
            scalar_cut = ((scalar.limbs_storage.limbs[m] >> m_shift)) & (unsigned)((pow(2, slice_bit) - 1));
        }
        else
        {
            /* Case: m and n are in the different scalar area index*/
            scalar_joint = ((uint128_t)scalar.limbs_storage.limbs[m]) | ((uint128_t)scalar.limbs_storage.limbs[n] << (Fr::LB));
            scalar_cut = ((unsigned)(scalar_joint >> m_shift)) & (unsigned)((pow(2, slice_bit) - 1));
        }

        /* Confirm the size of scalar_cut */
        if (scalar_cut > BIT_S_MASK)
        {
            printf("Error: [%s] scalar_cut out of BIT_S_MASK!\n", __FUNCTION__);
            return 1;
        }

        *slice = scalar_cut;

        return 0;
    }

    template <typename Fr>
    static unsigned calc_all_slices(unsigned **all_slices, Fr *scalars, unsigned msm_size, unsigned groups_number, unsigned bit_s, unsigned *slice_bit)
    {
        unsigned slice = 0;
        for (int i = 0; i < msm_size; i++)
        {
            /* Get slice of each group for one scalar*/
            for (uint32_t j = 0; j < groups_number; j++)
            {
                unsigned index_start = j * bit_s;
                unsigned index_end = (j < (groups_number - 1)) ? ((j + 1) * bit_s - 1) : (Fr::BC - 1);

                /* Get slice */
                int ret = get_slice(scalars[i], index_start, index_end, &slice, slice_bit[j]);

                if (ret)
                {
                    printf("Error: [%s] calc_bucket_index_area_27bit failed!\n", __FUNCTION__);
                    return 1;
                }
                all_slices[i][j] = slice;
            }
        }

        return 0;
    }

    template <typename Affine, typename Projective>
    cudaError_t init_buckets(Projective **&buckets, unsigned *buckets_num_in_each_group, unsigned groups_number, unsigned *slice_bit)
    {
        // as all the same bit
        unsigned bit_s = slice_bit[0];
        *buckets_num_in_each_group = (1 << bit_s) - 1;
        // unsigned buckets_num_in_all = buckets_num_in_each_group * groups_number;
        printf("[msm] bit_s: %d, groups_number: %d, buckets_num_in_each_group: %d\n", bit_s, groups_number, *buckets_num_in_each_group);

        HANDLE_RESULT_CUDA(cudaMallocHost((void **)&buckets, groups_number * sizeof(Projective *)));
        // cols
        for (int i = 0; i < groups_number; ++i)
        {
            HANDLE_RESULT_CUDA(cudaMallocHost((void **)&buckets[i], *buckets_num_in_each_group * sizeof(Projective)));
        }

        return cudaSuccess;
    }

    template <typename Affine, typename Projective, typename Fr>
    static unsigned aggregate_buckets(Projective **buckets, unsigned buckets_num_in_each_group, unsigned **h_all_slices, Affine *bases, unsigned msm_size, unsigned groups_number, const Affine *__restrict__ lookupTable, Projective *groups, bool lookup)
    {
        for (unsigned i = 0; i < groups_number; i++)
        {
            for (unsigned j = 0; j < msm_size; j++)
            {
                unsigned bucket_index = h_all_slices[j][i];

                /* Ignore index "0" */
                if (!bucket_index)
                {
                    continue;
                }

                if (lookup)
                {
                    groups[i] = groups[i] + lookupTable[(j << lookupBits) - j + bucket_index - 1];
                }
                else
                {
                    buckets[i][bucket_index - 1] = buckets[i][bucket_index - 1] + bases[j];
                }

                // print
                #if 0
                printf("    Madd compute count: %d, (%d, %d)\n", count, i, bucket_index);
                Projective::print(buckets[i][bucket_index - 1]);
                count++;
                #endif
            }
        }

        // for gpu debug
        #if 0
        for (unsigned i = 0; i < groups_number; i++)
        {
        unsigned buckets_this_group = i < groups_number - 1 ? buckets_num_in_each_group : (1 << Fr::BC % BIT_S) - 1;
        Projective groupResult = {0x0};
            for(unsigned j = 0; j < buckets_this_group; j++)
            {
                Projective result = {0x0};
                for(unsigned k = 0; k < j + 1; ++k)
                {
                    if(k == 1)
                        result = Projective::dbl(buckets[i][j]);
                    else
                        result = result + buckets[i][j];
                }

                groupResult = groupResult + result;
            }
            groups[i] = groupResult;
        }
        #endif

        return 0;
    }

    template <typename Projective>
    static unsigned calc_groups(Projective *groups, Projective **buckets, unsigned buckets_num_in_each_group, unsigned groups_number)
    {
        // unsigned count = 0;
        for (unsigned i = 0; i < groups_number; i++)
        {
            Projective running_sum = {0x0};
            Projective sum = {0x0};
            Projective tmp = {0x0};

            for (unsigned j = 0; j < buckets_num_in_each_group; j++)
            {
                // Note, reverse order
                running_sum = running_sum + buckets[i][buckets_num_in_each_group - 1 - j];
                sum = sum + running_sum;
            }
            groups[i] = sum;
        }

        return 0;
    }

    template <typename Projective>
    __forceinline__ static unsigned calc_groups_sums(Projective &result, Projective *groups, unsigned groups_number, unsigned *slice_bit)
    {
        Projective group_sum = {0x0};
        Projective g = {0x0};
        Projective g0 = groups[0];
        Projective tmp = {0x0};

        for (unsigned i = 0; i < (groups_number - 1); i++)
        {
            group_sum = group_sum + groups[groups_number - i - 1];

            for (unsigned j = 0; j < slice_bit[groups_number - i - 1 - 1]; j++)
            {
                group_sum = Projective::dbl(group_sum);
            }
        }

        result = group_sum + g0;
        return 0;
    }

    template <typename Fr>
    static void get_slice_bit(unsigned *slice_bit, unsigned groups_number, unsigned bit_s)
    {
        for (uint32_t j = 0; j < groups_number; j++)
        {
            unsigned index_start = j * bit_s;
            unsigned index_end = (j < (groups_number - 1)) ? ((j + 1) * bit_s - 1) : (Fr::BC - 1);
            slice_bit[j] = index_end - index_start + 1;
        }
    }

    template <typename Affine, typename Projective>
    void generateLookupTable(const Affine *__restrict__ bases, Affine *__restrict__ lookupTable, const unsigned msm_size, Projective *results)
    {
        constexpr unsigned lookupLen = (1 << lookupBits) - 1;
        constexpr Projective originP = {0x0};

        for (unsigned i = 0; i < msm_size; ++i)
        {
            Projective currentAP = originP + bases[i];
            Projective firstAP = currentAP;
            lookupTable[(i << lookupBits) - i] = Projective::to_affine(currentAP);
            lookupTable[(i << lookupBits) - i + 1] = Projective::to_affine(currentAP = Projective::dbl(currentAP));
            for (unsigned j = 2; j < lookupLen; ++j)
            {
                lookupTable[(i << lookupBits) - i + j] = Projective::to_affine(currentAP = currentAP + firstAP);
            }
        }
    }

    template <typename Fr, typename Affine, typename Projective>
    cudaError_t msm_execute_async_host(const execution_configuration<Fr, Affine, Projective> &exec_cfg, Projective **groups = nullptr, bool lookup = false)
    {
        cudaStream_t stream = exec_cfg.stream;
        Affine *bases = exec_cfg.bases;
        Fr *scalars = exec_cfg.scalars;
        Projective *results = exec_cfg.results;
        unsigned log_scalars_count = exec_cfg.log_scalars_count;
        unsigned msm_size = 1 << log_scalars_count;
        unsigned bit_s = get_bit_s();
        unsigned groups_number = get_group_number<Fr>(bit_s);

        Affine *lookupTable = nullptr;

// print
#if DEBUG_LOG
        printf("    log_scalars_count: %d, msm_size: %d, bit_s: %d, groups_number: %d\n", log_scalars_count, msm_size, bit_s, 
                groups_number);
        Affine::print(bases[0]);
        // Affine::print(bases[1]);
        Affine::print(bases[msm_size - 1]);
        Fr::print(scalars[0]);
        // Fr::print(scalars[1]);
        Fr::print(scalars[msm_size - 1]);
#endif

        for (int i = 0; i < msm_size; i++)
        {
            scalars[i] = Fr::from_montgomery(scalars[i]);   
        }

#if DEBUG_LOG
        printf("    scalar from_montgomery:\n");
        Fr::print(scalars[0]);
        Fr::print(scalars[msm_size - 1]);
#endif

        unsigned **h_all_slices = nullptr;
        HANDLE_RESULT_CUDA(cudaMallocHost((void **)&h_all_slices, msm_size * sizeof(unsigned *)));
        for (int i = 0; i < msm_size; ++i)
        {
            HANDLE_RESULT_CUDA(cudaMallocHost((void **)&h_all_slices[i], groups_number * sizeof(unsigned)));
        }

        unsigned h_slice_bit[groups_number];
        get_slice_bit<Fr>(h_slice_bit, groups_number, bit_s);

        Projective **h_buckets = nullptr;
        unsigned buckets_num_in_each_group = 0;
        init_buckets<Affine, Projective>(h_buckets, &buckets_num_in_each_group, groups_number, h_slice_bit);

        // calc slices
        calc_all_slices<Fr>(h_all_slices, scalars, msm_size, groups_number, bit_s, h_slice_bit);

        // calc buckets
        Projective *h_groups = new Projective[groups_number];
        memset(h_groups, 0, groups_number * sizeof(Projective));
        aggregate_buckets<Affine, Projective, Fr>(h_buckets, buckets_num_in_each_group, h_all_slices, bases, msm_size, groups_number, lookupTable, h_groups, lookup);

        // calc groups
        calc_groups<Projective>(h_groups, h_buckets, buckets_num_in_each_group, groups_number);

        unsigned long time1 = 0;
        struct timeval time_start, time_end;
        gettimeofday(&time_start, NULL);

        // dump
#if 0
        FILE *fp = fopen("./groups_sum_input_20.bin", "wb");
        for (int i = 0; i < groups_number; ++i)
        {
            fwrite((h_groups + i), sizeof(Projective), 1, fp);
            printf("group: %d\n", i);
            Projective::print(h_groups[i]);
            Projective::print(Projective::from_montgomery(h_groups[i]));
        }
        fclose(fp);
#endif

        //
        Projective h_result;
        calc_groups_sums<Projective>(h_result, h_groups, groups_number, h_slice_bit);
        printf("    gpu result: \n");
        Affine::print(Projective::to_affine(h_result));

        memcpy(results, &h_result, sizeof(Projective));

        // deinit
        for (int i = 0; i < msm_size; i++)
        {
            cudaFreeHost(h_all_slices[i]);
        }
        cudaFreeHost(h_all_slices);

        for (int i = 0; i < groups_number; i++)
        {
            cudaFreeHost(h_buckets[i]);
        }
        cudaFreeHost(h_buckets);

        delete[] h_groups;

        return cudaSuccess;
    }

    cudaError_t core_msm_execute_bn254_host(const msm_configuration configuration)
    {
        execution_configuration<CURVE_BN254::fr, CURVE_BN254::affine, CURVE_BN254::projective> cfg = {
            static_cast<cudaStream_t>(configuration.stream.handle),
            static_cast<CURVE_BN254::affine *>(configuration.bases),
            static_cast<CURVE_BN254::fr *>(configuration.scalars),
            static_cast<CURVE_BN254::projective *>(configuration.results),
            configuration.log_scalars_count
        };

        return msm_execute_async_host<CURVE_BN254::fr, CURVE_BN254::affine, CURVE_BN254::projective>(cfg);
    }
}
