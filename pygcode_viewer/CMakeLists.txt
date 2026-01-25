cmake_minimum_required(VERSION 3.15)
project(pygcode_viewer VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Suppress warnings for third-party code
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wno-unused-parameter -Wno-missing-field-initializers)
endif()

# Options
option(PYGCODE_BUILD_PYTHON "Build Python bindings" ON)
option(PYGCODE_BUILD_TESTS "Build tests" OFF)
option(PYGCODE_USE_SYSTEM_OPENGL "Use system OpenGL instead of bundled" ON)

# Find dependencies
find_package(OpenGL REQUIRED)

# Platform-specific OpenGL context libraries
if(UNIX AND NOT APPLE)
    # Linux: Use EGL for headless rendering
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(EGL REQUIRED egl)
    pkg_check_modules(GBM gbm)
    set(HEADLESS_LIBS ${EGL_LIBRARIES})
    set(HEADLESS_INCLUDES ${EGL_INCLUDE_DIRS})
    if(GBM_FOUND)
        list(APPEND HEADLESS_LIBS ${GBM_LIBRARIES})
        list(APPEND HEADLESS_INCLUDES ${GBM_INCLUDE_DIRS})
        add_compile_definitions(HAVE_GBM)
    endif()
    add_compile_definitions(PYGCODE_PLATFORM_LINUX)
elseif(APPLE)
    # macOS: Use CGL
    find_library(COCOA_LIBRARY Cocoa REQUIRED)
    find_library(IOKIT_LIBRARY IOKit REQUIRED)
    find_library(COREVIDEO_LIBRARY CoreVideo REQUIRED)
    find_library(OPENGL_LIBRARY OpenGL REQUIRED)
    set(HEADLESS_LIBS ${COCOA_LIBRARY} ${IOKIT_LIBRARY} ${COREVIDEO_LIBRARY} ${OPENGL_LIBRARY})
    set(HEADLESS_INCLUDES "")
    add_compile_definitions(PYGCODE_PLATFORM_MACOS)
elseif(WIN32)
    # Windows: Use WGL
    set(HEADLESS_LIBS opengl32)
    set(HEADLESS_INCLUDES "")
    add_compile_definitions(PYGCODE_PLATFORM_WINDOWS)
endif()

# libvgcode library (from PrusaSlicer)
add_subdirectory(extern/libvgcode)

# Core library
set(PYGCODE_SOURCES
    src/headless_context.cpp
    src/framebuffer_renderer.cpp
    src/camera.cpp
    src/config_parser.cpp
    src/gcode_parser.cpp
    src/gcode_viewer.cpp
)

set(PYGCODE_HEADERS
    src/headless_context.hpp
    src/framebuffer_renderer.hpp
    src/camera.hpp
    src/config_parser.hpp
    src/gcode_parser.hpp
    src/gcode_viewer.hpp
)

add_library(pygcode_core STATIC ${PYGCODE_SOURCES} ${PYGCODE_HEADERS})
target_include_directories(pygcode_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/libvgcode/include
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/libvgcode/glad/include
    ${HEADLESS_INCLUDES}
)
target_link_libraries(pygcode_core PUBLIC
    libvgcode
    OpenGL::GL
    ${HEADLESS_LIBS}
)

# Python bindings
if(PYGCODE_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    
    # Try to find pybind11
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        # Fetch pybind11 if not found
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
    
    pybind11_add_module(_pygcode_viewer src/bindings.cpp)
    target_link_libraries(_pygcode_viewer PRIVATE pygcode_core)
    
    # Install the Python module
    install(TARGETS _pygcode_viewer DESTINATION pygcode_viewer)
endif()

# Install headers for C++ usage
install(FILES ${PYGCODE_HEADERS} DESTINATION include/pygcode_viewer)
install(TARGETS pygcode_core DESTINATION lib)
