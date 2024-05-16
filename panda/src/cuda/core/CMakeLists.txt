
add_library(panda-cuda STATIC
        common/common.cu
        unit/ntt/fft.cu 
        unit/msm/msm_cuda.cuh
        curve/bn254/paramter.cuh
        panda_interface.cu
        panda_interface.cuh
        )

target_include_directories(panda-cuda PRIVATE "${PROJECT_SOURCE_DIR}/core")

set_target_properties(panda-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(panda-cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
install(TARGETS panda-cuda DESTINATION .)
