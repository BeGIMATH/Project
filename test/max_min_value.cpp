#include "image.h"

using namespace std;
using namespace cv;


int main(int argc, char** argv){

  Mat M = imread("../../data/clean_finger.png");
  Image I(M);
  cout << "The maximum pixel intesity is " << I.max_val() << endl;
  cout << "The minimum pixel intentity is " << I.min_val() << endl;
 
  
  return 0;
}
