cmake_minimum_required(VERSION 3.20)
project(taffy LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(e3 STATIC
  src/fft.cpp
  src/engine.cpp
)
target_include_directories(e3 PUBLIC include PRIVATE src)
target_compile_options(e3 PRIVATE -Wall -Wextra)

# Public API example CLI (examples/taffy_stretch.c)
add_executable(taffy_stretch examples/taffy_stretch.c)
target_link_libraries(taffy_stretch PRIVATE e3)
target_compile_options(taffy_stretch PRIVATE -Wall)
