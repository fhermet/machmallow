cmake_minimum_required(VERSION 3.24)
project(machmallow LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Metal backend: device/queue wrapper + metal-cpp implementation TU.
# Shaders are compiled at runtime (no Xcode required).
add_library(mm_metal STATIC
  src/backend/metal/MetalContext.cpp
  src/backend/metal/metal_impl.cpp
)
target_include_directories(mm_metal PUBLIC src third_party/metal-cpp)
target_link_libraries(mm_metal PUBLIC
  "-framework Metal"
  "-framework Foundation"
  "-framework QuartzCore"
)
target_compile_definitions(mm_metal PUBLIC
  MACHMALLOW_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

# Phase 0: Metal toolchain validation benchmark.
add_executable(machmallow src/main.cpp)
target_link_libraries(machmallow PRIVATE mm_metal)

# Phase 1: 1D Sod shock tube validation (CPU only).
add_executable(sod1d src/drivers/sod1d.cpp)
target_include_directories(sod1d PRIVATE src)

# Phase 2: 2D uniform-grid drivers (CPU only).
add_executable(sod2d src/drivers/sod2d.cpp)
target_include_directories(sod2d PRIVATE src)

add_executable(dmr src/drivers/dmr.cpp)
target_include_directories(dmr PRIVATE src)

# Phase 4: 2-level AMR validation on the Sod tube (CPU only).
add_executable(sod_amr src/drivers/sod_amr.cpp)
target_include_directories(sod_amr PRIVATE src)

# Phase 3: GPU step, validated against the CPU path.
add_executable(dmr_gpu src/drivers/dmr_gpu.cpp)
target_link_libraries(dmr_gpu PRIVATE mm_metal)

# Phase 5: hybrid CPU/GPU AMR on the double Mach reflection.
add_executable(dmr_amr src/drivers/dmr_amr.cpp)
target_link_libraries(dmr_amr PRIVATE mm_metal)

# Phase 7: viscous (Navier-Stokes) validation, CPU + GPU parity.
add_executable(shear src/drivers/shear.cpp)
target_link_libraries(shear PRIVATE mm_metal)

# Analytic validation suite (Toro battery, acoustic order, isentropic
# vortex, Sedov exponent).
add_executable(analytic_suite src/drivers/analytic_suite.cpp)
target_include_directories(analytic_suite PRIVATE src)

# v1.5 reacting flow: laminar deflagration (viscous reacting path).
add_executable(deflagration src/drivers/deflagration.cpp)
target_link_libraries(deflagration PRIVATE mm_metal)

# v1.5 reacting flow: 1D Chapman-Jouguet detonation (Strang coupling).
add_executable(detonation src/drivers/detonation.cpp)
target_link_libraries(detonation PRIVATE mm_metal)

# v1.5 reacting flow: 0D constant-volume reactor (reaction integrator).
add_executable(reactor src/drivers/reactor.cpp)
target_include_directories(reactor PRIVATE src)

# No-slip wall validation: Blasius flat-plate boundary layer (GPU).
add_executable(blasius src/drivers/blasius.cpp)
target_link_libraries(blasius PRIVATE mm_metal)

# GPU/hybrid vs pure-CPU throughput benchmark (study, run manually).
add_executable(benchmark src/drivers/benchmark.cpp)
target_link_libraries(benchmark PRIVATE mm_metal)

# Convergence study: observed L1 order of both schemes (smooth + Sod).
add_executable(convergence src/drivers/convergence.cpp)
add_executable(mms src/drivers/mms.cpp)
target_include_directories(mms PRIVATE src)

add_executable(cutcell_geom src/drivers/cutcell_geom.cpp)
target_include_directories(cutcell_geom PRIVATE src)

add_executable(cutcell_euler src/drivers/cutcell_euler.cpp)
target_include_directories(cutcell_euler PRIVATE src)

add_executable(immersed src/drivers/immersed.cpp)
target_include_directories(immersed PRIVATE src)

add_executable(immersed_case src/drivers/immersed_case.cpp)
target_include_directories(immersed_case PRIVATE src)

add_executable(immersed_amr src/drivers/immersed_amr.cpp)
target_include_directories(immersed_amr PRIVATE src)

add_executable(immersed_wedge src/drivers/immersed_wedge.cpp)
target_include_directories(immersed_wedge PRIVATE src)

add_executable(immersed_noslip src/drivers/immersed_noslip.cpp)
target_include_directories(immersed_noslip PRIVATE src)

add_executable(immersed_gpu src/drivers/immersed_gpu.cpp)
target_link_libraries(immersed_gpu PRIVATE mm_metal)
target_include_directories(convergence PRIVATE src)

# WENO5 + SSP-RK3 scheme validation (uniform grid, CPU).
add_executable(weno_suite src/drivers/weno_suite.cpp)
target_include_directories(weno_suite PRIVATE src)

# Two-gas (mass fraction + mixture gamma) scheme validation.
add_executable(species_suite src/drivers/species_suite.cpp)
target_include_directories(species_suite PRIVATE src)

# Multi-level AMR validation (CPU reference).
add_executable(ml_amr src/drivers/ml_amr.cpp)
target_include_directories(ml_amr PRIVATE src)

# Hybrid multi-level AMR validation + 3-level DMR showcase.
add_executable(mlgpu_amr src/drivers/mlgpu_amr.cpp)
target_link_libraries(mlgpu_amr PRIVATE mm_metal)

# Quantitative Haas & Sturtevant gate (two-gas helium bubble, GPU).
add_executable(hs_suite src/drivers/hs_suite.cpp)
target_link_libraries(hs_suite PRIVATE mm_metal)

# Velocity tagging + periodic AMR validation (Kelvin-Helmholtz).
add_executable(kh_amr src/drivers/kh_amr.cpp)
target_link_libraries(kh_amr PRIVATE mm_metal)

# Live view: programmatic Cocoa window + Metal render pass.
add_library(mm_render STATIC src/render/CocoaShim.mm)
target_include_directories(mm_render PUBLIC src)
set_source_files_properties(src/render/CocoaShim.mm PROPERTIES
  COMPILE_OPTIONS "-fobjc-arc")
target_link_libraries(mm_render PUBLIC "-framework Cocoa")

# Generic case runner driven by cases/*.ini files.
add_executable(run src/drivers/run.cpp)
target_link_libraries(run PRIVATE mm_metal mm_render)

# Declarative case system vs presets/exact solutions (+ gravity gates).
add_executable(casedef_test src/drivers/casedef_test.cpp)
target_link_libraries(casedef_test PRIVATE mm_metal)
