cmake_minimum_required(VERSION 3.16)

project(Snake_v2 LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(snake
    src/app/main.cpp
    src/engine/engine.cpp
)

target_compile_features(snake PRIVATE cxx_std_17)
set_target_properties(snake PROPERTIES CXX_EXTENSIONS OFF)

target_include_directories(snake PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(snake PRIVATE Threads::Threads)
