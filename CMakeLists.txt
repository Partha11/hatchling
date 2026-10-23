cmake_minimum_required(VERSION 3.16)

project(Snake_v2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_executable(snake
    src/app/main.cpp
    src/engine/engine.cpp
)

target_compile_features(snake PRIVATE cxx_std_17)
target_include_directories(snake PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(snake PRIVATE Threads::Threads)

# Python bindings via pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(snake_cpp
    src/bindings/bindings.cpp
    src/engine/engine.cpp
)

target_include_directories(snake_cpp PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_features(snake_cpp PRIVATE cxx_std_17)
