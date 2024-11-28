cmake_minimum_required(VERSION 3.10)
project(picograd)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directory for the shared library
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# Recursively collect all .cc files in the cpp-src directory
file(GLOB_RECURSE CPP_SOURCES ${CMAKE_SOURCE_DIR}/cpp-src/*.cc)

# Create the shared library
add_library(picograd SHARED ${CPP_SOURCES})

# Ensure Position Independent Code is enabled for shared library creation
set_target_properties(picograd PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Set platform-specific output name and suffix
if(APPLE)
    # On macOS, create a .dylib
    set_target_properties(picograd PROPERTIES SUFFIX ".dylib")
elseif(UNIX)
    # On Linux, create a .so
    set_target_properties(picograd PROPERTIES SUFFIX ".so")
elseif(WIN32)
    # On Windows, create a .dll
    set_target_properties(picograd PROPERTIES SUFFIX ".dll")
endif()

# Set debug flags (these are usually added automatically in Debug mode, but you can specify them explicitly)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Optionally, set the output name if needed (e.g., picograd)
set_target_properties(picograd PROPERTIES OUTPUT_NAME "picograd")
