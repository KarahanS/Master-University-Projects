cmake_minimum_required(VERSION 3.15)

project(
  Exercise4.1
  DESCRIPTION "Exercise 4.1: Amoeba Game"
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
add_library(submission SHARED submission/Amoeba.cpp submission/DeadCell.cpp submission/Food.cpp submission/Virus.cpp submission/Bacterium.cpp)
target_include_directories(submission PRIVATE lib/)
target_include_directories(submission PRIVATE include/)
target_link_libraries(submission PRIVATE project_options project_warnings)


# define the executable
add_executable(exercise_4_1 main.cpp)
# link the submission library
target_link_libraries(exercise_4_1 PRIVATE submission)
target_link_libraries(exercise_4_1 PRIVATE project_options project_warnings)
