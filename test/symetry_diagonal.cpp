#include "image.h"

using namespace std;
using namespace cv;


int main(int argc, char** argv){

  Mat M = imread("../../data/clean_finger.png");
  Image I(M);
  
  Mat M1 = I.symetry_diagonal();
  imwrite("../../data/symetry_diagonal.png", M1);
 
  return 0;
}
