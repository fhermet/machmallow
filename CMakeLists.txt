cmake_minimum_required(VERSION 3.24)
project(machmallow LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(machmallow
  src/main.cpp
  src/backend/metal/MetalContext.cpp
  src/backend/metal/metal_impl.cpp
)

target_include_directories(machmallow PRIVATE
  src
  third_party/metal-cpp
)

target_link_libraries(machmallow PRIVATE
  "-framework Metal"
  "-framework Foundation"
  "-framework QuartzCore"
)

# Shaders are compiled at runtime (no Xcode required) — point at the source tree.
target_compile_definitions(machmallow PRIVATE
  MACHMALLOW_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

# Phase 1: 1D Sod shock tube validation (CPU only).
add_executable(sod1d src/drivers/sod1d.cpp)
target_include_directories(sod1d PRIVATE src)

# Phase 2: 2D uniform-grid drivers (CPU only).
add_executable(sod2d src/drivers/sod2d.cpp)
target_include_directories(sod2d PRIVATE src)

add_executable(dmr src/drivers/dmr.cpp)
target_include_directories(dmr PRIVATE src)