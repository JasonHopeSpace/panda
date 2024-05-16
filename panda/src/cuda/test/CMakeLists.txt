find_package(CUDAToolkit)
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")
add_executable(panda_test
        msm_test.cu
        )

target_include_directories(panda_test PRIVATE "${PROJECT_SOURCE_DIR}/core")
target_link_libraries(panda_test panda-cuda)
target_link_libraries(panda_test gtest gtest_main CUDA::curand)

include(GoogleTest)
set_target_properties(panda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
gtest_discover_tests(panda_test)
