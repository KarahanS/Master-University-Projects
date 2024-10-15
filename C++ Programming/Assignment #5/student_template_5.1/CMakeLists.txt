cmake_minimum_required(VERSION 3.15)

project(
  Exercise5.1
  DESCRIPTION "Exercise 5.1: Word Histograms"
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
add_library(submission SHARED submission/load_books.cpp submission/histogram.cpp)
target_include_directories(submission PRIVATE lib/)
target_include_directories(submission PRIVATE include/)
target_link_libraries(submission PRIVATE project_options project_warnings)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(submission PRIVATE OpenMP::OpenMP_CXX)
endif()

# define the executable
add_executable(exercise_5_1 main.cpp)
# link the submission library
target_link_libraries(exercise_5_1 PRIVATE submission)
target_link_libraries(exercise_5_1 PRIVATE project_options project_warnings)

configure_file(FundamentalsOfBuisnesLaw.txt FundamentalsOfBuisnesLaw.txt COPYONLY)
configure_file(IntroductionToContracts.txt IntroductionToContracts.txt COPYONLY)
configure_file(PythonForEveryone.txt PythonForEveryone.txt COPYONLY)
configure_file(ThinkC++.txt ThinkC++.txt COPYONLY)
