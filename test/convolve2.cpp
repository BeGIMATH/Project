#include "image.h"

using namespace std;
using namespace cv;
using namespace Eigen;

int main(int argc, char** argv){


  Mat M = imread("../../data/clean_finger.png");
  Image I(M);
  int n = M.rows;
  int m = M.cols;

  
  Mat M1 = I.convolve2(9,9,2*n/3,m/2,80,60,1.0,1.5);
  imwrite("../../data/convolve2.png", M1);
  
  return 0;
}
