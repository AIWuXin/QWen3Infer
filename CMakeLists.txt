cmake_minimum_required(VERSION 3.22)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
# 自动检测CUDA路径
if(EXISTS /usr/local/cuda/bin/nvcc)
    set(CUDA_DIR "/usr/local/cuda")
    set(CMAKE_CUDA_COMPILER "${CUDA_DIR}/bin/nvcc")
else()
    # 尝试从环境变量或PATH查找
    find_program(CMAKE_CUDA_COMPILER nvcc PATHS ENV CUDACXX ENV CUDA_HOME ENV CUDA_ROOT)
    if(CMAKE_CUDA_COMPILER)
        get_filename_component(CUDA_BIN_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
        get_filename_component(CUDA_DIR ${CUDA_BIN_DIR} DIRECTORY)
    else()
        message(FATAL_ERROR "CUDA not found. Please install CUDA or set CUDACXX environment variable.")
    endif()
endif()
message(STATUS "CUDA found at: ${CUDA_DIR}")
project(QWen3Infer LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
    message(STATUS "CUDA debug flags enabled: -G -g")
endif()

include_directories(/usr/local/cuda-12.8/targets/x86_64-linux/include)
#include_directories(${CUDA_DIR}/include)
message(STATUS "Add CUDA Include at: ${CUDA_DIR}/include")
# Windows CUDA路径（仅用于参考）
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include

include(CheckSymbolExists)
check_symbol_exists(posix_memalign "stdlib.h" HAVE_POSIX_MEMALIGN)
if (HAVE_POSIX_MEMALIGN)
    add_compile_definitions(HAVE_POSIX_MEMALIGN)
endif ()

option(LLAMA3_SUPPORT OFF)   # Meta Llama3
option(QWEN2_SUPPORT OFF)    # Qwen2
option(QWEN3_SUPPORT OFF)    # Qwen3

if (LLAMA3_SUPPORT)
    message(STATUS "LLAMA3 SUPPORT")
    add_definitions(-DLLAMA3_SUPPORT)  # 向代码中注入宏定义
endif()

if (QWEN2_SUPPORT)
    message(STATUS "QWEN2 SUPPORT")
    add_definitions(-DQWEN2_SUPPORT)
endif()

if (QWEN3_SUPPORT)
    message(STATUS "QWEN3 SUPPORT")
    add_definitions(-DQWEN3_SUPPORT)
endif()

option(USE_CPM "Use CPM for dependency management" OFF)
if(USE_CPM)
    # ==========================================
    # 自动下载 CPM.cmake
    # ==========================================
    set(
            THIRD_PARTY_DIR
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    )
    set(
            CPM_DOWNLOAD_LOCATION
            "${THIRD_PARTY_DIR}/CPM.cmake"
    )
    if (NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
        message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
        if (NOT EXISTS ${THIRD_PARTY_DIR})
            file(
                    MAKE_DIRECTORY
                    ${THIRD_PARTY_DIR}
            )
        endif ()
        file(
                DOWNLOAD
                https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/CPM.cmake
                ${CPM_DOWNLOAD_LOCATION}
                SHOW_PROGRESS
        )
    endif ()

    # Use CPM to manage dependencies
    include(${CPM_DOWNLOAD_LOCATION})
    # 让 CPM 依赖不创建顶层目标（不在 CLion 中显示）
    set(CPM_USE_LOCAL_PACKAGES OFF)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)  # 允许选项被覆盖

#    CPMAddPackage(
#            NAME gflags
#            GITHUB_REPOSITORY gflags/gflags
#            VERSION 2.2.2
#            OPTIONS "GFLAGS_NAMESPACE google;INSTALL_SHARED_LIBS OFF"
#    )

    CPMAddPackage(
            NAME GTest
            GITHUB_REPOSITORY google/googletest
            VERSION 1.15.0
            EXCLUDE_FROM_ALL YES
    )

    CPMAddPackage(
            NAME glog
            GITHUB_REPOSITORY google/glog
            VERSION 0.7.1
            OPTIONS "BUILD_TESTING Off"
            EXCLUDE_FROM_ALL YES
    )

    CPMAddPackage(
            NAME Armadillo
            GITLAB_REPOSITORY conradsnicta/armadillo-code
            GIT_TAG 14.0.1
            EXCLUDE_FROM_ALL YES
    )

    CPMAddPackage(
            NAME sentencepiece
            GITHUB_REPOSITORY google/sentencepiece
            VERSION 0.2.0
            EXCLUDE_FROM_ALL YES
    )
    find_package(sentencepiece REQUIRED)

    if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
        CPMAddPackage(
                NAME absl
                GITHUB_REPOSITORY abseil/abseil-cpp
                GIT_TAG 20240722.0
                OPTIONS "BUILD_TESTING Off" "ABSL_PROPAGATE_CXX_STD ON" "ABSL_ENABLE_INSTALL ON"
        )
        CPMAddPackage(
                NAME re2
                GITHUB_REPOSITORY google/re2
                GIT_TAG 2024-07-02
        )
        CPMAddPackage(
                NAME nlohmann_json
                GITHUB_REPOSITORY nlohmann/json
                VERSION 3.11.3
        )
    endif()
endif()

add_compile_definitions(GLOG_USE_GLOG_EXPORT)

find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

file(
        GLOB_RECURSE DIR_BASE CONFIGURE_DEPENDS
        src/include/base/*.h
        src/source/base/*.cu
        src/source/base/*.cpp
)
file(
        GLOB_RECURSE DIR_TENSOR CONFIGURE_DEPENDS
        src/include/tensor/*.h
        src/source/tensor/*.cpp
)
file(
        GLOB_RECURSE DIR_SOURCES CONFIGURE_DEPENDS
        ${DIR_BASE}
        ${DIR_TENSOR}
)

add_executable(
        QWen3Infer
        hello.cu
        ${DIR_SOURCES}
)

target_link_libraries(
        QWen3Infer
        glog::glog
)

set_target_properties(QWen3Infer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# ==========================================
# 测试配置
# ==========================================
enable_testing()

# 收集测试文件
file(
        GLOB_RECURSE TEST_BASE CONFIGURE_DEPENDS
        test/test_base/*.cpp
        test/test_base/*.h
)
file(
        GLOB_RECURSE TEST_SRCS CONFIGURE_DEPENDS
        ${TEST_BASE}
)

if (TEST_BASE)
    add_executable(
            test_base
            ${DIR_SOURCES}
            ${TEST_BASE}
    )

    target_link_libraries(
            test_base
            glog::glog
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
    )
    set_target_properties(
            test_base PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME TestBase COMMAND test_base)
endif ()
