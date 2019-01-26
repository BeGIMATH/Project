#ifndef TOOLS_H
#define TOOLS_H
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

Eigen::MatrixXf cast_to_float(cv::Mat); /* A method which converts the image  to an int matrix to a float matrix*/

cv::Mat cast_to_int(Eigen::MatrixXf); /* A method which converts the image again to*/

Eigen::MatrixXf black_white_rec(Eigen::MatrixXf); /* A method to draw black and white rectangles in our image*/

Eigen::MatrixXf symetry_y_axis(Eigen::MatrixXf); /* A method to flip the image around the y-axis*/

Eigen::MatrixXf symetry_diagonal(Eigen::MatrixXf); /*A method to flip the image around the diagonal axis*/

float Euclidean_Distance(int x0, int y0, int x, int y ) /* A function which returns the Euclidean Distance between pixels*/

Eigen::MatrixXf isotropic(Eigen::MatrixXf M) /* A function which aproximates the weak_pressure modell*/

Eigen::MatrixXf asotrophic(Eigen::MatrixXf M) /*A function which aproximates the weak_pressure modeell*/

Eigen::MatrixXf Rotate_forward(Eigen::MatrixXf M, float theta) /* A method to rotate the image for the given angle theta, using the principle of forward mapping*/

Eigen::MatrixXf Rotate_backward(Eigen::MatrixXf M, float theta) /* A method to rotate the image for the given angle theta which is based on the backward  mapping principle */















#endif
