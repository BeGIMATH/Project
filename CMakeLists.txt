cmake_minimum_required(VERSION 3.1.0)

# Force project to be a C++ project
project(FingerPrint CXX)

# Force version of C++ standard
set (CMAKE_CXX_STANDARD 11)

# Required package for image processing
find_package(OpenCV REQUIRED)

# Required package for linear algebra
find_package(Eigen3 REQUIRED)

# Required package for fft
find_library(FFTW REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

#Bring the header, such as image.h into the project
include_directories(include)

#Manually adding the sources using the set command
# Creating the main library
add_subdirectory(src)

# Creating executable
add_subdirectory(demo)
