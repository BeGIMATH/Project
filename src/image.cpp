/**
 * @file image.cpp
 * @brief Image class implementation.
 */


#include "image.h" /* Image class definition */
#define EIGEN_FFTW_DEFAULT /* MACRO which makes possible to use FFT library */
#include <opencv2/opencv.hpp> /* Header file for opencv library */

#include "FFT" /* Header for the FFT Eigen library */

#include <iostream> /* std::cout */
#include <cmath> /* Use of functions \f$\cos (\theta )\f$, \f$\sin(\theta )\f$, power funtion and the square rootfunction. */


#define PI 3.1415926536 /*Approximate value of  number \f$\pi\f$ */
#define EMPTY 1.001 /* Used when working with Forward_Rotate method */



Image::Image(const cv::Mat  &M1):M(M1) {

}

 Eigen::MatrixXf Image::cast_to_float(cv::Mat M1){
    Eigen::MatrixXf m(M.rows,M.cols);
  for(int i=0;i<M.rows; i++){
    for(int j=0; j<M.cols; j++){
      m(i,j) = (int)M.at<uchar>(i,j,0);
      m(i,j) = m(i,j)/255;
    }
  }
  return m;
 }

 cv::Mat Image::cast_to_int(Eigen::MatrixXf m){
   cv::Mat A(m.rows(),m.cols(),CV_8UC1);
    for(int i=0;i<A.rows; i++){
      for(int j=0; j<A.cols; j++){
        int val = (int)(m(i,j)*255);
        A.at<uchar>(i,j,0) = (uchar)val;
      }
    }
    return A;
 }

int Image::min_val(){

    int Min= 0;
    for(int i=0;i<=M.cols;i++){
      for(int j=0;j<=M.rows;j++){

          Min = std::min(int(M.at<unsigned char>(i,j)),Min);

        }
      }
      return Min;
    }
int Image::max_val(){
  int Max = 0;
  for(int i=0;i<=M.cols;i++){
    for(int j=0;j<=M.rows;j++){

        Max = std::max(int(M.at<unsigned char>(i,j)),Max);

      }
    }
    return Max;
}



cv::Mat Image::black_white_rec(){
  Eigen::MatrixXf M1 = cast_to_float(M);

  for(int i=100; i<144; i++){
    for(int j=50; j<100; j++){
      M1(i,j) = 1;
    }
  }
  for(int i=200; i<250; i++){
    for(int j=180; j<220; j++){
      M1(i,j) = 0;
    }
  }
  cv::Mat M2 = cast_to_int(M1);
  return M2;
}

cv::Mat Image::symetry_y_axis(){
  Eigen::MatrixXf M1 = cast_to_float(M);

  int n = M1.rows();
  int m = M1.cols();

  Eigen::MatrixXf M2(n,m);

  for(int i=1; i<n; i++){
    for(int j=1; j<m; j++){
      M2(i,j) = M1(n-i,j);
    }
  }
  cv::Mat M3 = cast_to_int(M2);
  return M3;
}

cv::Mat Image::symetry_diagonal(){
  cv::Mat M1 = M.t();
  return M1;
}

float Image::Euclidean_Distance(int x0, int y0, int x, int y ) {
  float a ;
  a = sqrt((x0 - x)*(x0 - x) + 2*(y0 - y)*(y0 - y));
  return a;

}

cv::Mat Image::isotrophic(){

  Eigen::MatrixXf M1 = cast_to_float(M);

  int n = M1.rows();
  int m = M1.cols();

  int c_x = 5*n/7;
  int c_y = m/2;
  int s_x = 120;
  int s_y = 90;

   for(int i=1; i<n; i++){
    for(int j=1; j<m; j++){
      if((pow((c_x-i),2) + pow((c_y-j)*s_x/s_y,2)) > pow(s_x,2)){
          M1(i,j) = 1;
      }
    }
  }



  for(int i = 1 ; i < n; i++){
    for(int j = 1; j < m; j++){

      float r = Euclidean_Distance(c_x,c_y,i,j);
      M1(i,j) = 1-(1-M1(i,j))*exp(-pow(r/50,2));


    }
  }


  cv::Mat M2 = cast_to_int(M1);
  return M2;


}

cv::Mat Image::asotrophic(){
  Eigen::MatrixXf M1 = cast_to_float(M);
  int n = M1.rows();
  int m = M1.cols();

  int c_x = 5*n/7;
  int c_y = m/2;
  int s_x = 120;
  int s_y = 90;

  for(int i=1; i<n; i++){
    for(int j=1; j<m; j++){
      if((pow((c_x-i),2) + pow((c_y-j)*s_x/s_y,2)) > pow(s_x,2)){
          M1(i,j) = 1;
      }
    }
  }

  for(int i = n/2 - 143 ; i <= n/2 + 143; i++){
    for(int j = m/2 - 125; j <= m/2 + 125; j++){

      M1(i,j) = 1-(1-M1(i,j))*exp(-(pow(i-4*n/7,2)/2800 + pow(j-c_y,2)/1100));
    }
  }



  cv::Mat M2 = cast_to_int(M1);
  return M2;
}

float Image::middle(float x, float y, float Mx1y1, float Mx2y1, float Mx1y2, float Mx2y2){
  double dx, cx, dy, cy;
  dx = std::modf(x, &cx); // fractional part of $\ dx$
  dy = std::modf(y, &cy); //
  float res = (Mx1y1*(1-dx)*(1-dy) + Mx2y1*dx*(1-dy) + Mx1y2*(1-dx)*dy + Mx2y2*dx*dy) ;
  return res;
}






cv::Mat Image::Rotate_forward(float theta){
  Eigen::MatrixXf M1 = cast_to_float(M);

  float angle = theta*PI/180;
  int n = M1.rows();
  int m = M1.cols();

  Eigen::MatrixXf M2(n,m);

  int center_x = n/2;
  int center_y = m/2;


  for(int x=0; x<n; x++){
    for(int y=0;y<m; y++){
      M2(x,y) = EMPTY;
    }
  }

  for(int x=0; x<n; x++){
    for(int y=0;y<m; y++){
      float x2 = ((x - center_x) * cos(angle) - (y - center_y) * sin(angle))  + center_x ;
      float y2 = ((x - center_x) * sin(angle) + (y - center_y) * cos(angle))  + center_y;
      int xp = int(x2);
      int yp = int(y2);
      if((x2 > 0 && x2 < n-1 ) && (y2 > 0 && y2 < m-1)){

         M2(x,y) = middle(x2, y2, M1(xp,yp), M1(xp+1,yp), M1(xp,yp+1), M1(xp+1,yp+1));
      }
    }
  }

  cv::Mat M3 = cast_to_int(M2);
  return M3;
}









cv::Mat Image::Rotate_backward(float theta){

    Eigen::MatrixXf M1 = cast_to_float(M);
    float a = theta;
    float angle = a*PI/180;
    int n = M1.rows();
    int m = M1.cols();
    int center_x = n/2;
    int center_y = m/2;

    Eigen::MatrixXf M2(n,m);

    for(int x=0; x<n; x++){
      for(int y=0;y<m; y++){
        int xp = ((x - center_x) * cos(angle) + (y - center_y) * sin(angle)) + center_x ;
        int yp = ((-1)*(x - center_x) * sin(angle) + (y - center_y) * cos(angle)) + center_y;
          if((xp > 1 && xp < n-1 ) && (yp > 1 && yp < m-1)){

            M2(x,y) = M1(xp,yp);
          }
          else{
              M2(x,y) = EMPTY;
          }

      }
     }

    for(int i=1;i<n-1; i++){
    for(int j=1; j<m-1; j++){
      if (M2(i,j) == EMPTY) {

            M2(i,j) = (M2(i+1,j) + M2(i-1,j) + M2(i,j+1) + M2(i,j-1) + M2(i+1,j+1) + M2(i+1,j-1) + M2(i-1,j+1) + M2(i-1,j-1))/8;
            }
        }
     }



     cv::Mat M3 = cast_to_int(M2);
     return M3;
  }

  float Image::dist(int c_x, int c_y,int s_x,int s_y,int p1, int p2){
    return sqrt(pow((p1 - c_x)/s_x,2) + pow((p2 - c_y)/s_y,2)) - 1;
}

double Image::brake(float t, float k){
  if(t<=0) {return 0;}
  if(t<=k) {return  (1-cos(t*PI/k))/2;}
  if(t>k) {return 1;}
}

cv::Mat  Image::distortion(int c_x, int c_y, int s_x, int s_y,float angle,float dx, float dy,float k){

    Eigen::MatrixXf M1 = cast_to_float(M);
    Eigen::MatrixXf M2 = Eigen::MatrixXf::Ones(M1.rows(),M1.cols());


    float theta = angle*(PI/180);

    int n = M1.rows();
    int m = M1.cols();
    int deltax;
    int deltay;

    for(int i = 1 ; i <= n-1; i++){
      for(int j = 1; j <= m-1; j++){
        if((pow((c_x-i),2) + pow((c_y-j)*s_x/s_y,2)) < pow(s_x,2)){
          float sh = dist(c_x,c_y,s_x,s_y,i,j);
          deltax= int((cos(theta)*(i-c_x) + sin(theta)*(j-c_y)) + c_x + dx - i);
          deltay= int((cos(theta)*(j-c_y) - sin(theta)*(i-c_x)) + c_y + dy - j);

          int xp = int(i + deltax*(1-brake(sh,k)));
          int yp = int(j + deltay*(1-brake(sh,k)));

          if((xp > 1 && xp < n-1 ) && (yp > 1 && yp < m-1)){

            M2(i,j) = M1(xp,yp);
            }

           }
        else{
          M2(i,j) = M1(i,j);
        }

        }
      }
      cv::Mat M3 = cast_to_int(M2);
      return M3;
  }

  cv::Mat  Image::convolve2D(Eigen::MatrixXf K){
    Eigen::MatrixXf M1 = cast_to_float(M);

    Eigen::MatrixXf M2 = Eigen::MatrixXf::Zero(M1.rows(),M1.cols());
    int dataSizeX = M1.cols();
    int dataSizeY = M1.rows();
    int i, j, m, n, mm, nn;


    float sum;
    int ii,jj;

    int kCenterX = int(K.cols()/2);
    int kCenterY = int(K.rows()/2);

    for(i=0; i < M1.rows(); i++)
    {
        for(j=0; j < M1.cols(); j++)
        {
            sum = 0;
            for(m=0; m < K.rows(); m++)
            {
                mm = K.rows() - 1 - m;

                for(n=0; n < K.cols(); n++)
                {
                    nn = K.cols() - 1 - n;


                     ii = (i + (kCenterY - mm));
                     jj = (j + (kCenterX - nn));


                    if(ii >= 0 && ii < M1.rows() && jj >= 0 && jj < M1.cols()){
                        M2(i,j) += M1(ii,jj) * K(mm,nn);





                    }



                }
            }

        }
    }

    cv::Mat M3 = cast_to_int(M2);
    return M3;


  }

  cv::Mat Image::conv_fft(Eigen::MatrixXf K){
   Eigen::MatrixXf M1 = cast_to_float(M);
   Eigen::MatrixXf K1 = Eigen::MatrixXf::Zero(M1.rows(),M1.cols());
    for (int i=0; i < K.rows(); i++){
        for(int j=0; j < K.cols(); j++){
            K1(i,j) = K(i,j);
        }
    }
    int n = M1.rows();
    int m = M1.cols();

    Eigen::MatrixXcf M2(n,m);
    Eigen::MatrixXcf K2(n,m);
    Eigen::FFT< float > fft;
    for (int k = 0; k< n; k++){
        Eigen::VectorXcf tmpOut(n);
        Eigen::VectorXcf tmpOut1(n);
        fft.fwd(tmpOut1, K1.row(k));
        fft.fwd(tmpOut, M1.row(k));
        M2.row(k) = tmpOut;
        K2.row(k) = tmpOut1;

    }
    Eigen::FFT< float > fft2;
    for (int k=0; k < m; k++){
        Eigen::VectorXcf tmpOut(m);
        Eigen::VectorXcf tmpOut1(m);
        fft2.fwd(tmpOut1,K2.col(k));
        fft2.fwd(tmpOut, M2.col(k));
        M2.col(k) = tmpOut;
        K2.col(k) = tmpOut1;
    }
    Eigen::MatrixXcf C(n,m);
    C = M2.cwiseProduct(K2);


    Eigen::MatrixXcf C1(n,m);

    Eigen::FFT< float > ifft;
    for (int k = 0; k < m; k++) {
        Eigen::VectorXcf tmpOut(m);
        ifft.inv(tmpOut, C.col(k));
        C1.col(k) = tmpOut;
    }

    Eigen::FFT< float > ifft2;
    for(int k = 0; k < n; k++){
        Eigen::VectorXcf tmpOut(n);
        ifft2.inv(tmpOut, C1.row(k));
        C1.row(k) = tmpOut;
    }

    cv::Mat M3 = cast_to_int(C1.real());
    return M3;
}



cv::Mat Image::convolve2(int k_height, int k_width, int c_x, int c_y, int s_x, int s_y, double inner_sigma, double outer_sigma){



  Eigen::MatrixXf M1 = cast_to_float(M);


  Eigen::MatrixXf kernel = getGaussian(k_height, k_width, inner_sigma);
  Eigen::MatrixXf kernel1= getGaussian(k_height, k_width, outer_sigma);

  cv::Mat C = conv_fft(kernel);
  cv::Mat D = conv_fft(kernel1);
  Eigen::MatrixXf C1 = cast_to_float(C);
  Eigen::MatrixXf D1 = cast_to_float(D);




  for (int i=0 ; i<M1.rows() ; i++) {
      for (int j=0 ; j<M1.cols() ; j++) {
        if((pow(c_x-i,2) + pow((c_y-j)*s_x/s_y,2)) > pow(s_x,2)){

           M1(i,j) = C1(i,j);
          }
        else{

          M1(i,j) = D1(i,j);
        }
      }
  }
  cv::Mat M2 = cast_to_int(M1);
  return M2;

}

Eigen::MatrixXf Image::getGaussian(int height, int width, double sigma){
  Eigen::MatrixXf kernel(height, width);
  int kcenter=int(height/2);
    double sum=0.0;

    double s=2.0*sigma*sigma;
    for (int i=0 ; i<width ; i++) {
        for (int j=0 ; j<height ; j++) {
            double r=sqrt((i-kcenter)*(i-kcenter)+(j-kcenter)*(j-kcenter));
            kernel(i,j) = (exp(-(r*r)/s))/(PI*s);
            sum += kernel(i,j);
        }
    }

for (int i=0 ; i<width ; i++) {
        for (int j=0 ; j<height ; j++) {
            kernel(i,j) /= sum;
}


    }
    return kernel;
}

Eigen::MatrixXf Image::getBoxBlur(int height, int width){
  Eigen::MatrixXf kernel(height, width);
  int sum=0;
  int i,j;
  for(i=0; i<width; i++){
    for(j=0; j<height; j++){
      kernel(i,j)=1;
      sum += kernel(i,j);
    }
  }

    for (i=0 ; i<width ; i++) {
            for (j=0 ; j<height ; j++) {
                kernel(i,j) /= sum;
             }
            }
        return kernel;
}

cv::Mat Image::dry(){
  Eigen::MatrixXf M1 = cast_to_float(M);

  int n = M1.rows();
  int m = M1.cols();

  float b = 0.5;
  for(int x=0; x<n; x++){
    for(int y=0;y<m; y++){
      M1(x,y) = int( pow(float(M1(x,y)),b)*255 ) / 255.0;
    }
  }
   cv::Mat M2 = cast_to_int(M1);
  return M2;
}

cv::Mat Image::moist(){
  Eigen::MatrixXf M1 = cast_to_float(M);

  int n = M1.rows();
  int m = M1.cols();

  float b = 2.0;
  for(int x=0; x<n; x++){
    for(int y=0;y<m; y++){

      M1(x,y) = int( pow(float(M1(x,y)),b)*255 ) / 255.0;
    }
  }
   cv::Mat M2 = cast_to_int(M1);
  return M2;
}
