cmake_minimum_required(VERSION 3.15)

project(
  TemplateFunctions
  DESCRIPTION "Exercise 7 - Templates"
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

# NOTE: this exercise does not require writing something to a .cpp file, so no submission library is built.
# Define the library that is compiled from the submission
#add_library(submission SHARED submission/exercise_07.cpp)
#target_link_libraries(submission PRIVATE project_options project_warnings)

# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#   target_link_libraries(submission PRIVATE OpenMP::OpenMP_CXX)
# endif()

# define the executable
add_executable(exercise_07 main.cpp)
# link the submission library
# target_link_libraries(exercise_07 PRIVATE submission)
target_link_libraries(exercise_07 PRIVATE project_options project_warnings)
