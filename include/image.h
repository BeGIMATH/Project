/** 
 * @file image.h
 * @brief Image class declaration
*/
#ifndef IMAGE_H
#define IMAGE_H
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

/**
 * @class Image
 * @brief Object representing an image, declared of type Mat.
 */
class Image{
  private:
    /** 
     * @brief Private atribute an image, usually declared as Mat type.
     */
    cv::Mat M;     
  public:
    /**
     * @brief Image constructor.
     * @param  M1 a cv::Mat type.
     */  
    Image(const cv::Mat &M1); 
    
    /** 
     * @brief A method to convert a cv::Mat type to Eigen::MatrixXf type.
     * @brief This method maps [0,255] to [0,1] ,by dividing each number of that interval by 255. 
     */
    Eigen::MatrixXf cast_to_float(cv::Mat); 

    /** @brief A method to convert a Eigen::MatrixXf type to cv::Mat type.
     *  @brief It maps [0 ,1] to [0,255], by multiplying each number in of that intervale by 255. 
     *  So this  method is the inverse of the other one, [0,1] to [0,255].
     */
    cv::Mat cast_to_int(Eigen::MatrixXf m); 

    /**
     *   @brief A method to find the maximum pixel intensity.
     *   @return maximum pixel intensity.
     */
    int max_val();

    /** 
     * @brief This finds the  minimum pixel intensity. 
     * @return minimum pixel intensity.
     */
    int min_val(); 
    
    /** 
     *@brief This method is used to drwtwo rectangles.
     * First one filled with  white color and the second one filled with black color. 
     */
    cv::Mat black_white_rec();
    
    /**
     *  @brief  Rotates the image around the horizontal axis.
     */
    cv::Mat symetry_y_axis(); 

    /** 
     * @brief Flip the image around the diagonal axis, so around the line \f$ y=x \f$
     */
    cv::Mat symetry_diagonal(); 

    /** 
     * @return The Euclidean Distance between pixels, pixels considered as points.
     * @param  x0 -first coordinate of the first point
     * @param  y0 -second coordinate of the first point.
     * @param  x  -first coordinate of the second point.
     * @param  y  -second coordinate of the  second point.
     */
    float Euclidean_Distance(int x0, int y0, int x, int y ); 

    /**
     *  @brief This method tries to approximate the given  weak_pressure model using the isotrophic principle. 
     */
    cv::Mat isotrophic(); 

     /** 
      * @brief this method tries to approximate the weak_pressure model using the asotrophic principle.
      */
    cv::Mat asotrophic();

    /** 
     * @brief This method is used to calculate the weighted average between four pixels.
     * @param x  - The first coordinate of the pixel we want to calculate the weighted average.
     * @param y  - The second coordinate of the pixel we want to calculate the weighted average. 
     * @param Mx1y1 - The first neighbour of hte given pixel.
     * @param Mx2y1 - The second neighbour of the given pixel.
     * @param Mx1y2 - The third neighbour of the given pixel.
     * @param Mx2y2 - The fourth neighbot of the given pixel.
     * @return the weighted average pixel intensity.
     */
    float middle(float x, float y, float Mx1y1, float Mx2y1, float Mx1y2, float Mx2y2);

    /** 
     *  @brief A method to rotate the image, which is based on the forward  mapping principle.
     *  @brief It also returns the new coordinates of the rotated pixels.
     *  @param angle of rotation \f$\theta\f$.
     *  @brief Rotation is done around the center of image.
     */
    cv::Mat Rotate_forward(float angle); 

    /** @brief A method to rotate the image, which is based on the backward  mapping principle.
     *  @brief It also returns the new coordinates of the rotated pixels.
     *  @param angle of rotation \f$\theta\f$.
     *  @brief Rotation is done around the center of image.
     */
    cv::Mat Rotate_backward(float angle); 
    
    /** 
     * @brief A method calculate the eliptic distance between to pixels substracted by one.
     * @param c_x -first coordinate of the center of elipse.
     * @param c_y -second coordinate of the center of elipse.
     * @param s_x -first semiaxis of the elipse.
     * @param s_y -second semiaxis of the elipse.
     * @param p1  -first coordinate of the point for which we are going to calculate the distance from the center pixel.
     * @param p2  -second coordinate of the point for which we are going to calculate the distance from the center pixel.
     */
    float dist(int c_x, int  c_y,int s_x,int s_y,int p1, int p2);

    /** 
     * @brief Simple function needed to simulate the motion inside a region in a fingerprint image.
     * @param t -the argument of function.
     * @param k -the elestacity coefficient;
     */
    double brake(float t, float k); 

    /** 
     * @brief With this method we try to aproximate the fingerprint warping model.
     * @param c_x     -first coordinate of the center of elipse.
     * @param c_y     -second coordinate of the center of elipse.
     * @param s_x     -first semiaxis of the elipse.
     * @param s_y     -second semiaxis of the elipse.
     * @param angle   -angle of rotation.
     * @param dx      -first coordinate of the translation vector with the begin point the original pixel.
     * @param dy      -seconf coordinate of the translation vector with the begining point the original pixel.
     * @param k       -coefficient of elasticity.
     */
    cv::Mat distortion(int c_x, int c_y, int s_x, int s_y,float angle,float dx, float dy,float k); 
    
    /**
     * @brief A simple  method which returns the convolution for two matrices.
     * @param K
     */ 
    cv::Mat convolve2D(Eigen::MatrixXf K); 
    
    /** 
     * @brief A simple  method which returns the convolution for two matrices but it uses the efficiency of fft.
     * @param K
     */ 
    cv::Mat conv_fft(Eigen::MatrixXf K); 
  
    /** 
     * @brief A method which returns the Gaussian kernel for given dimmensions of that matrix.
     * @param height  -the number of the rows of the matrix.
     * @param width  -the number of columns of the matrix.
     * @param sigma -parameter which affect the gaussian function.
     * @return Gaussian Kernel.
     */
    Eigen::MatrixXf getGaussian(int height, int width, double sigma); /* A method which returns the Gaussian Kernel for given dimensions and parameter sigma*/

    /** 
     * @brief A method which returns the Box-blur kernel for given dimmensions of that matrix.
     * @param height the number of the rows of the matrix.
     * @param width the number of columns of the matrix.
     * @return the BoxBlur Kernel.
     */  
    Eigen::MatrixXf getBoxBlur(int height, int width); /* A method which returns the box_blur kernel for given dimensions*/
    
    /* @brief A method to do the convolution of an image,but for a choosen eliptical region it gives different
     * @brief results inside and outside that region based on the values of the $\sigma$ parameters.
     * @param height       -the number of the rows of the matrix.
     * @param width        -the number of columns of the matrix.
     * @param c_x          -first coordinate of the center of eliptical region.
     * @param c_y          -second coordinate of the center of eliptical region.
     * @param s_x          -first semiaxis of the eliptical region.
     * @param s_y          -second semiaxis of the eliptical region.
     * @param inner_sigma  -first parameter for the gaussion kernel, choosen for the interior of the region.
     * @param outer_sigma  -second parameter for the gaussian kernel, choosen for the exterior of the region.
     */
    cv::Mat convolve2(int k_height, int k_width, int c_x, int c_y, int s_x, int s_y, double inner_sigma, double outer_sigma);
    /* @brief A method to simulate the dry finger artefact.
     */
    cv::Mat dry();

     /* @brief A method to simulate the moist finger artefact.
     */
    cv::Mat moist();
};

#endif /* End of include guard: IMAGE_H */
