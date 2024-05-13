
#include <cuda_runtime.h>
#include "field/field.cuh"
#include "curve/bn254/config.cuh"

#include "fft.cuh"


constexpr unsigned LOG2_MAX_ELEMENTS = 32;
constexpr unsigned MAX_LOG2_RADIX = 8;
constexpr unsigned MAX_LOG2_LOCAL_WORK_SIZE = 7;

namespace panda_ntt
{
    template <class Field>
    __device__ void field_pow(Field* in_out, Field base, unsigned exponent)
    {
        #if 0
        in_out->limbs_storage = in_out->get_one();

        while(exponent > 0) 
        {
            if (exponent & 1)
            {
                *in_out *= base;
            }
            exponent = exponent >> 1;

            if(exponent == 0)
            {
                break;
            }
            base ^= 2;
        }
        #endif
    }

    template <class Field>
    __global__ void fft_init_omegas(Field* omegas)
    {
        if(threadIdx.x < LOG2_MAX_ELEMENTS)
        {
            //omegas[threadIdx.x + 1] = omegas[0].pow_vartime(1 << (threadIdx.x + 1));
            if(threadIdx.x > 0)
            {
                field_pow(&omegas[threadIdx.x], omegas[0], 1 << (threadIdx.x));
            }
        }
    }

    template <class Field>
    __global__ void fft_init_twiddle(Field* twiddles, Field* omegas, unsigned max_deg, unsigned n)
    {
        if(threadIdx.x < (1 << max_deg >> 1))
        {
            Field twiddle = {};
            field_pow(&twiddle, omegas[0], n >> max_deg);
            field_pow(&twiddles[threadIdx.x], twiddle, threadIdx.x);
        }
    }

    template <class Field> 
    cudaError_t set_up(Field* omega, Field** omegas)
    {
        cudaError_t errorcode = cudaMalloc(omegas, LOG2_MAX_ELEMENTS * sizeof(Field));
        errorcode = cudaMemcpy ( (void*)*omegas, (void*)omega, sizeof(Field), cudaMemcpyHostToDevice );
        
        // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
        fft_init_omegas<<< 1, LOG2_MAX_ELEMENTS >>>(*omegas);

        errorcode = cudaStreamSynchronize(0);
        return errorcode;
    }

    __device__ inline unsigned bitreverse(unsigned n, unsigned bits) 
    {
        unsigned r = 0;
        for(unsigned i = 0; i < bits; i++) 
        {
            r = (r << 1) | (n & 1);
            n >>= 1;
        }
        return r;
    }

    template <class Field>
    __device__ void field_pow_lookup(Field* in_out, Field* bases, unsigned exponent)
    {
        #if 0
        in_out->limbs_storage = in_out->get_one();
        unsigned i = 0;
        while(exponent > 0) 
        {
            if (exponent & 1)
            {
                *in_out *= bases[i];
            }
            exponent = exponent >> 1;
            i++;
        }
        #endif
    }

    /*
    * FFT algorithm is inspired from: http://www.bealto.com/gpu-fft_group-1.html
    */
    template <class Field>
    __global__ void radix_fft(  Field* x, // Source buffer
                                Field* y, // Destination buffer
                                Field* pq, // Precalculated twiddle factors
                                Field* omegas,
                                unsigned n, // Number of elements
                                unsigned lgp, // Log2 of `p` (Read more in the link above)
                                unsigned deg, // 1=>radix2, 2=>radix4, 3=>radix8, ...
                                unsigned max_deg) // Maximum degree supported, according to `pq` and `omegas`
    {
        #if 0
        extern  __shared__ Field u[];

        unsigned p = 1 << lgp;
        unsigned k = blockIdx.x & (p - 1);

        x += blockIdx.x;
        y += ((blockIdx.x - k) << deg) + k;

        unsigned count = 1 << deg; // 2^deg
        unsigned counth = count >> 1; // Half of count

        unsigned counts = count / blockDim.x * threadIdx.x;
        unsigned counte = counts + count / blockDim.x;

        // Compute powers of twiddle
        Field twiddle = {};
        field_pow_lookup(&twiddle, omegas, (n >> lgp >> deg) * k);
        Field tmp = {};
        field_pow(&tmp, twiddle, counts);
        for(unsigned i = counts; i < counte; i++) 
        {
            u[i] = tmp * x[i * (n >> deg)];
            tmp *= twiddle;
        }
        __syncthreads();

        const unsigned pqshift = max_deg - deg;
        for(unsigned rnd = 0; rnd < deg; rnd++) 
        {
            const unsigned bit = counth >> rnd;
            for(unsigned i = counts >> 1; i < counte >> 1; i++) 
            {
                const unsigned di = i & (bit - 1);
                const unsigned i0 = (i << 1) - di;
                const unsigned i1 = i0 + bit;
                tmp = u[i0];
                //u[i0] += u[i1];
                u[i0] = u[i0] + u[i1];
                u[i1] = tmp - u[i1];
                if(di != 0) u[i1] = pq[di << rnd << pqshift] * u[i1];
            }

            __syncthreads();
        }

        for(unsigned i = counts >> 1; i < counte >> 1; i++) 
        {
            y[i*p] = u[bitreverse(i, deg)];
            y[(i+counth)*p] = u[bitreverse(i + counth, deg)];
        }
        #endif
    }

    template <class Field> 
    cudaError_t execute(execution_configuration<Field> &exec_cfg)
    {

        unsigned n = 1 << exec_cfg.log_n;
        // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
        unsigned max_deg = min(MAX_LOG2_RADIX, exec_cfg.log_n);

        // Precalculate:
        // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
        Field* twiddles = nullptr;// Precalculated twiddle factors
        cudaError_t errorcode = cudaMalloc(&twiddles, sizeof(Field) * (1 << max_deg >> 1));
        fft_init_twiddle<<< 1, 1 << max_deg >> 1 >>>(twiddles, exec_cfg.omegas, max_deg, n);
        errorcode = cudaStreamSynchronize(0);

        // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
        unsigned log_p = 0;

        unsigned loopTimes = 0;
        Field* tmp = nullptr;

        // Each iteration performs a FFT round
        while (log_p < exec_cfg.log_n) 
        {
            // 1=>radix2, 2=>radix4, 3=>radix8, ...
            unsigned deg = min(max_deg, exec_cfg.log_n - log_p);

            unsigned local_work_size = 1 << min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
            unsigned global_work_size = n >> deg;

            radix_fft<<<global_work_size, local_work_size, (1 << deg) * sizeof(Field), 0>>>(exec_cfg.d_src, exec_cfg.d_dst, twiddles, exec_cfg.omegas, n, log_p, deg, max_deg);
            errorcode = cudaStreamSynchronize(0);

            tmp = exec_cfg.d_dst;
            exec_cfg.d_dst = exec_cfg.d_src;
            exec_cfg.d_src = tmp;

            loopTimes++;
            log_p += deg;
        }
        *exec_cfg.flag = loopTimes & 0x01;

        errorcode = cudaFree(twiddles);

        return errorcode;
    }

    cudaError_t tear_down()
    {
        return cudaSuccess;
    }

    CURVE_BN254::fr* omegas = nullptr;// [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]

    cudaError_t core_ntt_setup_bn254(void *input_omega)
    {
        CURVE_BN254::fr *omega = (CURVE_BN254::fr *)input_omega;
        return set_up<CURVE_BN254::fr>(omega, &omegas);
    }

    cudaError_t core_ntt_execute_bn254(ntt_configuration configuration)                       
    {
        execution_configuration<CURVE_BN254::fr> cfg = {  static_cast<cudaMemPool_t>(configuration.mem_pool.handle),
                                                        static_cast<cudaStream_t>(configuration.stream.handle),
                                                        static_cast<CURVE_BN254::fr *>(configuration.d_src),
                                                        static_cast<CURVE_BN254::fr *>(configuration.d_dst),
                                                        omegas,
                                                        configuration.log_n,
                                                        static_cast<unsigned *>(configuration.flag)};

        return execute<CURVE_BN254::fr>(cfg);
    }

    cudaError_t core_ntt_execute_bn254_v1(panda_ntt_configuration_v1 configuration)                       
    {

        CURVE_BN254::fr *omega = static_cast<CURVE_BN254::fr *>(configuration.d_omega);
        CURVE_BN254::fr* omegas = nullptr;

        set_up<CURVE_BN254::fr>(omega, &omegas);
        execution_configuration<CURVE_BN254::fr> cfg = {  static_cast<cudaMemPool_t>(configuration.mem_pool.handle),
                                                        static_cast<cudaStream_t>(configuration.stream.handle),
                                                        static_cast<CURVE_BN254::fr *>(configuration.d_src),
                                                        static_cast<CURVE_BN254::fr *>(configuration.d_dst),
                                                        omegas,
                                                        configuration.log_n,
                                                        static_cast<unsigned *>(configuration.flag)};

        return execute<CURVE_BN254::fr>(cfg);
    }

    cudaError_t core_ntt_tear_down()
    {
        return tear_down();
    }

}// namespace panda_ntt