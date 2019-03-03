cmake_minimum_required(VERSION 3.1.0)
project(FingerPrint)

# Required package for image processing
find_package( OpenCV 3.2.0 REQUIRED)

# Required package for linear algebra
find_package( Eigen3 3.3 REQUIRED NO_MODULE)


#Bring the header, such as image.h into the project
include_directories(include)

# Creating the main library
add_subdirectory(src)

#Generate tests
enable_testing()
add_subdirectory(test)

#Generate documentation
add_subdirectory(doc)
