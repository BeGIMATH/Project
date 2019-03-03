#include "image.h"

using namespace std;
using namespace cv;


int main(int argc, char** argv){

  Mat M = imread("../../data/clean_finger.png");
  Image I(M);
  int n = M.rows;
  int m = M.cols;

  
  Mat M1 = I.distortion(n*2/3,m/2,70,60,40.0,20.0,20.0,1.0);
  imwrite("../../data/distortion.png", M1);
  
  return 0;
}
