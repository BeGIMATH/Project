#Generating an executable file

add_executable(main_exe main.cpp)

target_link_libraries( main_exe ${OpenCV_LIBS} ${Eigen3_LIBS}  ${fftw3} FingerPrint)
