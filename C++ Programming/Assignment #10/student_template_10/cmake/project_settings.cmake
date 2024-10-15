# Add new build types
message(STATUS "Adding build type AddressSanitized")
set(CMAKE_CXX_FLAGS_ADDRESSSANITIZED
    "-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1"
    CACHE STRING "Flags used by the C++ compiler during sanitized builds."
          FORCE)
set(CMAKE_C_FLAGS_ADDRESSSANITIZED
    "-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1"
    CACHE STRING "Flags used by the C compiler during sanitized builds."
          FORCE)
set(CMAKE_EXE_LINKER_FLAGS_ADDRESSSANITIZED
    "-fsanitize=address,undefined -g"
    CACHE STRING "Flags used for linking binaries during coverage builds."
          FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_ADDRESSSANITIZED
    "-fsanitize=address,undefined -g"
    CACHE STRING
          "Flags used by the shared libraries linker during sanitized builds."
          FORCE)
mark_as_advanced(
  CMAKE_CXX_FLAGS_ADDRESSSANITIZED CMAKE_C_FLAGS_ADDRESSSANITIZED
  CMAKE_EXE_LINKER_FLAGS_ADDRESSSANITIZED
  CMAKE_SHARED_LINKER_FLAGS_ADDRESSSANITIZED)

# Set default build type if non was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "AddressSanitized")
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)

# Use ccache if available
find_program(CCACHE ccache)
if(CCACHE)
    message(STATUS "Using ccache")
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
ENDIF(CCACHE)

# GENERATE COMPILE_COMMANDS.JSON
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add options for link time optimizations (LTO) aka interprocedural optimization
option(ENABLE_IPO
    "Enable interprocedural optimizations" OFF)
if(ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if(result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(SEND_ERROR "IPO is not supported: ${output}")
    endif()
endif()
