
cmake_minimum_required(VERSION 3.0.1)

add_executable(max_min max_min_value.cpp)
target_link_libraries(max_min ${OpenCV_LIBS} ${Eigen3_LIBS}   FingerPrint)

add_executable(black_white_rec black_white_rec.cpp)
target_link_libraries(black_white_rec ${OpenCV_LIBS} ${Eigen3_LIBS}   FingerPrint)

add_executable(symetry_y symetry_y.cpp)
target_link_libraries(symetry_y ${OpenCV_LIBS} ${Eigen3_LIBS}   FingerPrint)

add_executable(symetry_diagonal symetry_diagonal.cpp)
target_link_libraries(symetry_diagonal ${OpenCV_LIBS} ${Eigen3_LIBS}   FingerPrint)

add_executable(isotrophic isotrophic.cpp)
target_link_libraries(isotrophic ${OpenCV_LIBS} ${Eigen3_LIBS}   FingerPrint)

add_executable(asotrophic asotrophic.cpp)
target_link_libraries(asotrophic ${OpenCV_LIBS} ${Eigen3_LIBS}   FingerPrint)

add_executable(rotate_forward rotate_forward.cpp)
target_link_libraries(rotate_forward ${OpenCV_LIBS} ${Eigen3_LIBS}  FingerPrint)

add_executable(rotate_backward rotate_backward.cpp)
target_link_libraries(rotate_backward ${OpenCV_LIBS} ${Eigen3_LIBS}  FingerPrint)


add_executable(distortion distortion.cpp)
target_link_libraries(distortion ${OpenCV_LIBS} ${Eigen3_LIBS}  FingerPrint)

add_executable(convolve2D convolve2D.cpp)
target_link_libraries(convolve2D ${OpenCV_LIBS} ${Eigen3_LIBS}  FingerPrint)

add_executable(convolve_fft convolve_fft.cpp)
target_link_libraries(convolve_fft ${OpenCV_LIBS} ${Eigen3_LIBS} ${fftw}  FingerPrint)

add_executable(convolve2 convolve2.cpp)
target_link_libraries(convolve2 ${OpenCV_LIBS} ${Eigen3_LIBS} ${fftw}  FingerPrint)

add_executable(dry dry.cpp)
target_link_libraries(dry ${OpenCV_LIBS} ${Eigen3_LIBS}   FingerPrint)

add_executable(moist moist.cpp)
target_link_libraries(moist ${OpenCV_LIBS} ${Eigen3_LIBS}  FingerPrint)


add_test (NAME max_min
    COMMAND max_min
    )
add_test (NAME black_white_rec
    COMMAND black_white_rec
    )
add_test (NAME symetry_y
    COMMAND symetry_y
    )
add_test (NAME symetry_diagonal
    COMMAND symetry_diagonal
    )
add_test (NAME isotrophic
    COMMAND isotrophic
    )
add_test (NAME asotrophic
    COMMAND asotrophic
    )
add_test (NAME rotate_forward
    COMMAND rotate_forward
    )
add_test (NAME rotate_backward
    COMMAND rotate_backward
    )
add_test (NAME distortion
    COMMAND distortion
    )
add_test (NAME convolve2D
    COMMAND convolve2D
    )
add_test (NAME convolve_fft
    COMMAND convolve_fft
    )
add_test (NAME convolve2
    COMMAND convolve2
    )
add_test (NAME dry
    COMMAND dry
    )
add_test (NAME moist
    COMMAND moist
    )