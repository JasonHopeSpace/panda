#pragma once


namespace panda_cuda
{

    #define DEBUG_LOG 0

    #define PANDA_ASM 1
    #define SWITCH_32_64 1
    #define PANDA_ASM_32 SWITCH_32_64
    #define PANDA_ASM_64 !SWITCH_32_64

#if PANDA_ASM
    #if PANDA_ASM_32
        #define FIELD_BITS 32
        #define uint_bits uint32_t
    #elif PANDA_ASM_64
        #define FIELD_BITS 64
        #define uint_bits uint64_t
    #endif
#else
    #define FIELD_BITS 64
    #define uint_bits uint64_t
#endif
}