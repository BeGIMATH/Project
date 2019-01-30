/*!\file image.h
* \brief Image class declaration
*/
#ifndef IMAGE_H
#define IMAGE_H
//#include "tools.h"

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

/**
 * \class Image
 * \brief Object representing an image, declared of type Mat.
 */
class Image{
  private:
     
    
    cv::Mat M; /* Private atribute an image, usually declared as Mat type*/
    
  public:
    /*!
     * \brief Image constructor
     * \param a cv::Mat type 
     */  
    Image(const cv::Mat &); 
    
    /*! \brief Convert a cv::Mat type to Eigen::MatrixXf type.
     *  \brief It maps $\left[0,255 right ] to left[ 0,1 right]$ 
     */
    Eigen::MatrixXf cast_to_float(cv::Mat); 

    /*! \brief Convert a Eigen::MatrixXf type to cv::Mat type.
     *  \brief It maps $\left[0 ,1 right ] to left[ 0,255 right]$ 
     */
    cv::Mat cast_to_int(Eigen::MatrixXf m); /*A method to conver back [0,1] to [0,255] */
    
    /*!  \brief Find the minimum pixel intensity. */
    int max_val();
    /*! \brief Find the minimum pixel intensity. */
    int min_val(); 
    
    /*! \brief Draw two rectangles, first one white color and the second one black color. */
    cv::Mat black_white_rec();
    
    /*! \brief Flip the image around the y-axis*/
    cv::Mat symetry_y_axis(); 

    /*! \brief Flip the image around the diagonal axis*/
    cv::Mat symetry_diagonal(); 

    /*! \brief Returns the Euclidean Distance between pixels*/

    float Euclidean_Distance(int x0, int y0, int x, int y ); 

    /*! \brief Aproximate the weak_pressure model using the isotrophic principle. */
    cv::Mat isotrophic(); 

     /*! \brief Aproximate the weak_pressure model using the asotrophic principle*/
    cv::Mat asotrophic();

    /*! \brief A method to rotate the image, which is based on the forward  mapping principle.
     *  \brief It also returns the new coordinates of the rotated pixels.
     *  \param Angle of rotation $\theta$.
     *  \brief Rotation is done around the center of image.
     */
    cv::Mat Rotate_forward(float theta); 

    /*! \brief A method to rotate the image, which is based on the backward  mapping principle.
     *  \brief It also returns the new coordinates of the rotated pixels.
     *  \param Angle of rotation $\theta$.
     *  \brief Rotation is done around the center of image.
     */
    cv::Mat Rotate_backward(float theta); 
    
    /*! \brief Calculate the eliptic distance substracted by one.
     * \param c_x -first coordinate of the center of elipse.
     * \param c_y -second coordinate of the center of elipse.
     * \param s_x -irst semiaxis of the elipse.
     * \param s_y -second semiaxis of the elipse.
     * \param p1  -first coordinate of the point for which we are going to calculate the distance from the center pixel.
     * \param p2  -second coordinate of the point for which we are going to calculate the distance from the center pixel.
     */
    float dist(int c_x, int  c_y,int s_x,int s_y,int p1, int p2); /* A function which return the eliptical distance */

    /*! \brief Simple function needed to simulate the motin inside a region in a fingerprint image.
     *  \param $\ t$-the argument of function.
     *  \param $\ k$-the elestacity coefficient;
     */
    double brake(float t, float k); 

    /*! \brief Simulate the fingerprint warping.
     * \param c_x -first coordinate of the center of elipse.
     * \param c_y -second coordinate of the center of elipse.
     * \param s_x -irst semiaxis of the elipse.
     * \param s_y -second semiaxis of the elipse.
     * \param p1  -first coordinate of the point for which we are going to calculate the distance from the center pixel.
     * \param p2  -second coordinate of the point for which we are going to calculate the distance from the center pixel.
     * \param angle -angle of rotation.
     * \param $\ dx$-first coordinate of the translation vector with the begin point the original pixel.
     * \param $\ dy$-seconf coordinate of the translation vector with the begining point the original pixel.
     * \param $\ k$ -coefficient of elasticity.
     */
    cv::Mat distortion(int c_x, int c_y, int s_x, int s_y,float angle,float dx, float dy,float k); 
    
    /*! \brief A simple  method which returns the convolution for two matrices.
     *  \param Kernel $\ K$ 
     */ 
    cv::Mat convolve2D(Eigen::MatrixXf K); 
    
    /*! \brief A simple  method which returns the convolution for two matrices but it uses the efficiency of fft.
     *  \param Kernel $\ K$ 
     */ 
    cv::Mat conv_fft(Eigen::MatrixXf K); /* A method which returns the convolution for two matrices, but using the power of fft */
    
    /*! \brief A method which returns the Gaussian kernel for given dimmensions of that matrix.
     *  \param $\ height$ the number of the rows of the matrix.
     *  \param $\ width$ the number of columns of the matrix.
     */
    Eigen::MatrixXf getGaussian(int height, int width, double sigma); /* A method which returns the Gaussian Kernel for given dimensions and parameter sigma*/

    /*! \brief A method which returns the Box-blur kernel for given dimmensions of that matrix.
     *  \param $\ height$ the number of the rows of the matrix.
     *  \param $\ width$ the number of columns of the matrix.
     */  
    Eigen::MatrixXf getBoxBlur(int height, int width); /* A method which returns the box_blur kernel for given dimensions*/
    
    /*  \brief A method to do the convolution of an image,but for a choosen eliptical region it gives different
     *  \breif results inside and outside that region based on the values of the $\sigma$ parameters.
     *  \param $\ height$ the number of the rows of the matrix.
     *  \param $\ width$ the number of columns of the matrix.
     *  \param c_x -first coordinate of the center of eliptical region.
     *  \param c_y -second coordinate of the center of eliptical region.
     *  \param s_x -irst semiaxis of the eliptical region.
     *  \param s_y -second semiaxis of the eliptical region.
     *  \param $\sigam$1 -first parameter for the gaussion kernel, choosen for the interior of the region.
     *  \param $\sigma$2 -second parameter for the gaussian kernel, choosen for the exterior of the region.
     */
    cv::Mat convolve2(int k_height, int k_width, int c_x, int c_y, int s_x, int s_y, double inner_sigma, double outer_sigma);/* A method which returns which applies different blurring result for a given region and outside that region*/
    
};

#endif /* End of include guard: IMAGE_H */

