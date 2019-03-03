#include "image.h"

using namespace std;
using namespace cv;


int main(int argc, char** argv){

  Mat M = imread("../../data/clean_finger.png");
  Image I(M);
  
  Mat M1 = I.symetry_y_axis();
  imwrite("../../data/symetry_y.png", M1);
 
  return 0;
}
