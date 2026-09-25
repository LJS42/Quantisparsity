cmake_minimum_required(VERSION 3.20)
project(QuantizationSparse VERSION 0.1.0 LANGUAGES CXX)

option(QUANTIZATION_SPARSE_BUILD_SOLVER "Build the CUDA experiment and CUDA tests" ON)
option(QUANTIZATION_SPARSE_BUILD_TESTS "Build/register experiment tests" ON)
set(QUANTIZATION_SPARSE_WEIGHTS "576" CACHE STRING "Weights per row (excluding indicators)")
if(NOT QUANTIZATION_SPARSE_WEIGHTS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "QUANTIZATION_SPARSE_WEIGHTS must be a positive integer")
endif()

if(QUANTIZATION_SPARSE_BUILD_SOLVER)
    # The caller selects CMAKE_CUDA_ARCHITECTURES; no library source paths are used.
    enable_language(CUDA)
    find_package(GPUBoscia 0.1 CONFIG REQUIRED)
    find_package(CUDAToolkit REQUIRED)

    function(quantization_sparse_add_executable target source weights)
        add_executable(${target} ${source})
        target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_link_libraries(${target} PRIVATE gpuboscia::gpuboscia CUDA::cublas)
        target_compile_definitions(${target} PRIVATE QUANTIZATION_SPARSE_WEIGHTS=${weights})
        set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endfunction()
    quantization_sparse_add_executable(quantization_sparse src/main.cu ${QUANTIZATION_SPARSE_WEIGHTS})
endif()

if(QUANTIZATION_SPARSE_BUILD_TESTS)
    enable_testing()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME exported_rows
        COMMAND "${Python3_EXECUTABLE}" -m unittest discover
            -s "${CMAKE_CURRENT_SOURCE_DIR}/experiments" -p "test_*.py" -v)
    if(QUANTIZATION_SPARSE_BUILD_SOLVER)
        quantization_sparse_add_executable(test_quantization tests/test_quantization.cu ${QUANTIZATION_SPARSE_WEIGHTS})
        add_test(NAME quantization_candidates COMMAND test_quantization)
        # Exhaustive checks use four weights rather than a 576-variable search.
        quantization_sparse_add_executable(test_quantization_small src/main.cu 4)
        add_test(NAME quantization_exhaustive
            COMMAND test_quantization_small --k 2 --samples 16 --seed 42 --time-limit 60 --verify)
        add_test(NAME quantization_zero_budget
            COMMAND test_quantization_small --k 0 --samples 16 --seed 42 --time-limit 60 --verify)
    endif()
endif()
