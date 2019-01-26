#include "image.h"
#include "tools.h"
using namespace std;
using namespace cv;
using namespace Eigen;

int main(int argc, char** argv){

  Mat M = imread("../data/clean_finger.png");
  Image I(M);
  cout << "The maximum pixel intesity is " << I.max_val() << endl;
  cout << "The minimum pixel intentity is " << I.min_val() << endl;
  
  MatrixXd M1 =  cast_to_float(M);
  Mat M2 = cast_to_int(M1);


  imwrite("test.png", M2);
  namedWindow("test", WINDOW_NORMAL);
  imshow("test", M2);
  waitKey( 0 );
  destroyWindow("test");
  return 0;
}
