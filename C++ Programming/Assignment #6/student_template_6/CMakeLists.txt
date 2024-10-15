cmake_minimum_required(VERSION 3.15)

project(
  CSG
  DESCRIPTION "Constructive Solid Geometry"
  LANGUAGES CXX)

# We include some common default settings (release type...) and other cmake
# flags that can be set to optimize or speed up the build if the required tools
# can be found.
include(cmake/project_settings.cmake)

# We use the `project_options` library declarations to define some compile time
# options.
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_20)

# warning specifications are defined in the `project_warnings` library.
add_library(project_warnings INTERFACE)
include(cmake/compiler_warnings.cmake)
set_project_warnings(project_warnings)

# Define the library that is compiled from the submission
add_library(submission SHARED include/point3d.h
                              include/aabb.h
                              include/shapes_test.h
                              submission/shapes.cpp
                              submission/shapes.h
                              submission/transformations.cpp
                              submission/transformations.h
                              submission/operations.cpp
                              submission/operations.h
                              submission/voxel_grid.cpp
                              submission/voxel_grid.h
                              submission/your_shape.cpp)
target_include_directories(submission PRIVATE submission/)
target_include_directories(submission PRIVATE include/)
target_link_libraries(submission PRIVATE project_options project_warnings)

# TODO: enable these to test parts of your implementation locally (requires libstdc++-10 or libc++-13)
target_compile_definitions(submission PUBLIC TEST_TASK_1)
target_compile_definitions(submission PUBLIC TEST_TASK_2)
target_compile_definitions(submission PUBLIC TEST_TASK_3)
target_compile_definitions(submission PUBLIC TEST_TASK_4)

# TODO: enable this to see some example output (this will not compile until your implementation is complete)
target_compile_definitions(submission PUBLIC ENABLE_DEMO_CODE)

# silence some shadowing warnings
target_compile_options(submission PUBLIC "-Wno-shadow-field-in-constructor")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(submission PRIVATE OpenMP::OpenMP_CXX)
endif()

# define the executable
add_executable(csg main.cpp)
# link the submission library
target_link_libraries(csg PRIVATE submission)
target_link_libraries(csg PRIVATE project_options project_warnings)
target_include_directories(csg PRIVATE include/)
target_include_directories(csg PRIVATE submission/)

