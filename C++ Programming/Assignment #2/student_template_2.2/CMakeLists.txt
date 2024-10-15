cmake_minimum_required(VERSION 3.15)

project(
  InsertionSort
  DESCRIPTION "This is the student template for exercise2.2"
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
add_library(submission SHARED submission/insertion_sort.h
                              submission/insertion_sort.cpp)
target_link_libraries(submission PRIVATE project_options project_warnings)
target_include_directories(submission PRIVATE submission/)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(submission PRIVATE OpenMP::OpenMP_CXX)
endif()

# define the executable
add_executable(insertion_sort main.cpp)
# link the submission library
target_link_libraries(insertion_sort PRIVATE submission)
target_link_libraries(insertion_sort PRIVATE project_options project_warnings)
