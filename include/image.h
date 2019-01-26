#ifndef IMAGE_H
#define IMAGE_H

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>



class Image{
  private:
    
    cv::Mat M; /* Private atribute an image, usually declared as Mat type*/
  public:

    
    Image(const cv::Mat &); /*Constructor*/

    cv::Mat get_image(); /* Accessor*/

    int max_val(); /* Return maximum pixel intensity value*/
    int min_val(); /* Return minimum pixel intensity value*/



};

#endif
