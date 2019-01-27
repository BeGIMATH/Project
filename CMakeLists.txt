cmake_minimum_required(VERSION 3.1.0)
project(FingerPrint)

# Required package for image processing
find_package( OpenCV 3.2.0 REQUIRED)

# Required package for linear algebra
find_package( Eigen3 3.3 REQUIRED NO_MODULE)

# Required package for fft
find_library(FFTW REQUIRED)

#Bring the header, such as image.h into the project
include_directories(include)

#Manually adding the sources using the set command
#set(SOURCES src/main.cpp src/image.cpp src/tools.cpp)
# Creating the main library
add_subdirectory(src)

# Creating executable
add_subdirectory(demo)
