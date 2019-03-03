#include "image.h"

using namespace std;
using namespace cv;
using namespace Eigen;

int main(int argc, char** argv){

  Mat M = imread("../../data/clean_finger.png");
  Image I(M);
  MatrixXf K = I.getGaussian(9,9,1);
  Mat M1 = I.conv_fft(K);
  imwrite("../../data/convolve_fft.png", M1);
  
  return 0;
}
