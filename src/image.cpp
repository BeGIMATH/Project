#include "image.h"

using namespace std;
using namespace cv;
using namespace Eigen;



Image::Image(const cv::Mat  &M1):M(M1) {
  
}



int Image::min_val(){

    int Min= 0;
    for(int i=0;i<=M.cols;i++){
      for(int j=0;j<=M.rows;j++){

          Min = min(int(M.at<unsigned char>(i,j)),Min);

        }
      }
      return Min;
    }
int Image::max_val(){
  int Max = 0;
  for(int i=0;i<=M.cols;i++){
    for(int j=0;j<=M.rows;j++){

        Max = max(int(M.at<unsigned char>(i,j)),Max);

      }
    }
    return Max;
}

Mat Image::get_image(){
  return M;
}



