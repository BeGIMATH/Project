#include "tools.h"

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/FFT>
#include <fftw3.h>
#include <iostream>
#include <cmath>

#define PI 3.1415926536

using namespace std;
using namespace Eigen;
using namespace cv;



Eigen::MatrixXf cast_to_float(cv::Mat M){
  Eigen::MatrixXf m(M.rows,M.cols);
  for(int i=0;i<M.rows; i++){
    for(int j=0; j<M.cols; j++){
      m(i,j) = (int)M.at<uchar>(i,j,0);
      m(i,j) = m(i,j)/255;
    }
  }
  return m;
}

cv::Mat cast_to_int(Eigen::MatrixXf m) {
    cv::Mat A(m.rows(),m.cols(),CV_8UC1);
    for(int i=0;i<A.rows; i++){
      for(int j=0; j<A.cols; j++){
        int val = (int)(m(i,j)*255);
        A.at<uchar>(i,j,0) = (uchar)val;
      }
    }
    return A;
  }

Eigen::MatrixXf black_white_rec(Eigen::MatrixXf M){
  for(int i=30; i<60; i++){
    for(int j=15; j<50; j++){
      M(i,j) = 1;
    }
  }
  for(int i=200; i<250; i++){
    for(int j=180; j<220; j++){
      M(i,j) = 0;
    }
  }
  return M;

}


Eigen::MatrixXf symetry_y_axis(Eigen::MatrixXf M){
  Eigen::MatrixXf M1 = M;
  for(int i=0; i<M.rows(); i++){
    for(int j=0; j<M.cols(); j++){
      M1(i,j) = M(M.rows() - i,j);
    }
  }
  return M1;
}

Eigen::MatrixXf symetry_diagonal(Eigen::MatrixXf M){
  return M.transpose();
}

Eigen::MatrixXf isotropic(Eigen::MatrixXf M){

}
Eigen::MatrixXf asotrophic(Eigen::MatrixXf M){

}

float Euclidean_Distance(int x0, int y0, int x, int y ) {
  float a ;
  a = sqrt((x0 - x)*(x0 - x) + 2*(y0 - y)*(y0 - y));
  return a;

}

Eigen::MatrixXf Rotate_forward(Eigen::MatrixXf M, float theta){
 /* Some work to do with interpolation an out of bounds pixels*/
}

Eigen::MatrixXf Rotate_backward(Eigen::MatrixXf M, float theta){
  Eigen::MatrixXf M1(M.rows(),M.cols());

    float a = theta;
    float angle = a*PI/180;
    int n = M.rows();
    int m = M.cols();
    int center_x = n/2;
    int center_y = m/2;

    float scale = 1/sqrt(1+float(m)/float(n));


    

    for(int x=0; x<n; x++){
      for(int y=0;y<m; y++){
        int xp = ((x - center_x) * cos(angle) + (y - center_y) * sin(angle)) + center_x ;
        int yp = ((-1)*(x - center_x) * sin(angle) + (y - center_y) * cos(angle)) + center_y;
          if((xp > 1 && xp < n-1 ) && (yp > 1 && yp < m-1)){
            //M1(x,y) = (M(xp+1,yp+1)+M(xp+1,yp-1)+M(xp-1,yp+1)+M(xp-1,yp-1))/4;
            M1(x,y) = M(xp,yp);
          }
      }
     }

    for(int x=1; x<n-1; x++){
      for(int y=1;y<m-1; y++){
             int xp = ((x - center_x) * cos(angle) + (y - center_y) * sin(angle)) + center_x ;
             int yp = ((-1)*(x - center_x) * sin(angle) + (y - center_y) * cos(angle)) + center_y;
             if((xp < 0 && xp > n-1 ) && (yp > 0 && yp < m-1)){
               M1(x,y) = (M1(x+1,y+1)+M1(x+1,y-1)+M(x-1,y+1)+M(x-1,y-1))/4;
             }
           }
         }

    return M1;
  }





float dist(int c_x, int c_y,int s_x,int s_y,int p1, int p2){
  return sqrt(pow((p1 - c_x)/s_x,2) + pow((p2 - c_y)/s_y,2)) - 1;
}

double brake(float t, float k){
  if(t<=0) {return 0;}
  if(t<=k) {return  (1-cos(t*PI/k))/2;}
  if(t>k) {return 1;}
}

Eigen::MatrixXd distortion(Eigen::MatrixXf M,int c_x, int c_y, int s_x, int s_y,float angle,float dx, float dy,float k){
  float theta = angle*(PI/180);
    MatrixXd M1 = MatrixXd::Ones(M.rows(),M.cols());
    int n = M.rows();
    int m = M.cols();
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
          //if((pow((c_x-xp)/s_x,2) + pow((c_y-yp)/s_y,2)) <= 1){
            M1(i,j) = M(xp,yp);
          }
        }
        else {
          M1(i,j) = M(i,j);
        }



      }
    }
    return M1;
  }

Eigen::MatrixXf convolve2D(Eigen::MatrixXf M, Eigen::MatrixXf K){
  Eigen::MatrixXf M1 = Eigen::MatrixXf::Zero(M.rows(),M.cols());
    int dataSizeX = M.cols();
    int dataSizeY = M.rows();
    int i, j, m, n, mm, nn;
    
                        
    float sum;                                      
    int ii,jj;

    int kCenterX = int(K.cols()/2);
    int kCenterY = int(K.rows()/2);

    for(i=0; i < M.rows(); i++)                // rows
    {
        for(j=0; j < M.cols(); j++)            // columns
        {
            sum = 0;                            // init to 0 before sum
            for(m=0; m < K.rows(); m++)      // kernel rows
            {
                mm = K.rows() - 1 - m;       // row index of flipped kernel

                for(n=0; n < K.cols(); n++)  // kernel columns
                {
                    nn = K.cols() - 1 - n;   // column index of flipped kernel

                    // index of input signal, used for checking boundary
                     ii = (i + (kCenterY - mm));
                     jj = (j + (kCenterX - nn));

                    // ignore input samples which are out of bound
                    if(ii >= 0 && ii < M.rows() && jj >= 0 && jj < M.cols()){
                        M1(i,j) += M(ii,jj) * K(mm,nn);
                        /*if( M(i,j) > 1){
                            M(i,j) -= M(ii,jj)*K(mm,nn);
                            
                        }
                        */
                        
                        
                        

                    }
                    
                    

                }
            }
            
        }
    }

    return M1 ;
}

Eigen::MatrixXf conv_fft(Eigen::MatrixXf M,Eigen::MatrixXf K){
   Eigen::MatrixXf K1 = Eigen::MatrixXf::Zero(M.rows(),M.cols());
    for (int i=0; i < K.rows(); i++){
        for(int j=0; j < K.cols(); j++){
            K1(i,j) = K(i,j); // Zero padding the kernel k
        }
    }
    int n = M.rows();
    int m = M.cols();
    
    Eigen::MatrixXcf M1(n,m);
    Eigen::MatrixXcf K2(n,m);
    Eigen::FFT< float > fft;
    for (int k = 0; k< n; k++){
        Eigen::VectorXcf tmpOut(n);
        Eigen::VectorXcf tmpOut1(n);
        fft.fwd(tmpOut1, K1.row(k));
        fft.fwd(tmpOut, M.row(k));
        M1.row(k) = tmpOut;
        K2.row(k) = tmpOut1;

    }
    Eigen::FFT< float > fft2;
    for (int k=0; k < m; k++){
        Eigen::VectorXcf tmpOut(m);
        Eigen::VectorXcf tmpOut1(m);
        fft2.fwd(tmpOut1,K2.col(k));
        fft2.fwd(tmpOut, M1.col(k));
        M1.col(k) = tmpOut;
        K2.col(k) = tmpOut1;
    }
    Eigen::MatrixXcf C(n,m);
    C = M1.cwiseProduct(K2);
    
    
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
    Eigen::MatrixXf A(n,m);
    return A = C1.real();
}

Eigen::MatrixXf getGaussian(int height, int width, double sigma){
  Eigen::MatrixXf kernel(height, width);
  int kcenter=int(height/2);
    double sum=0.0;
    int i,j;
    double s=2.0*sigma*sigma;
    for (i=0 ; i<width ; i++) {
        for (j=0 ; j<height ; j++) {
            double r=sqrt((i-kcenter)*(i-kcenter)+(j-kcenter)*(j-kcenter));
            kernel(i,j) = (exp(-(r*r)/s))/(PI*s);
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

Eigen::MatrixXf getBoxBlur(int height, int width){
  Eigen::MatrixXf kernel(height, width);
  int sum=0;
  int i,j;
  for(i=0; i<width; i++){
    for(j=0; j<height; j++){
      kernel(i,j)=1;
      sum += kernel(i,j);
    }}

    for (i=0 ; i<width ; i++) {
            for (j=0 ; j<height ; j++) {
                kernel(i,j) /= sum;
  }}
            return kernel;
          }

float dist(int c_x, int c_y,int s_x,int s_y,int p1, int p2){
  return sqrt(pow((p1 - c_x)/s_x,2) + pow((p2 - c_y)/s_y,2)) - 1;
}

Eigen::MatrixXf convolve2(Eigen::MatrixXf M,int k_height, int k_width, int c_x, int c_y, int s_x, int s_y, double inner_sigma, double outer_sigma){
  Eigen::MatrixXf kernel(k_height,k_width);
  Eigen::MatrixXf kernel1(k_height,k_width);

  Eigen::MatrixXf convoluted(M.rows(), M.cols());

  kernel = getGaussian(k_height, k_width, inner_sigma);
  kernel1=getGaussian(k_height, k_width, outer_sigma);

  Eigen::MatrixXf C(M.rows(),M.cols());
  Eigen::MatrixXf D(M.rows(),M.cols());

  C = conv_fft(M,kernel);
  D = conv_fft(M,kernel1);
  
  
  int i,j;
  for (i=0 ; i<M.rows() ; i++) {
      for (j=0 ; j<M.cols() ; j++) {
        if((pow(c_x-i,2) + pow((c_y-j)*s_x/s_y,2)) < pow(s_x,2)){
           //kernel = getGaussian(k_height, k_width, inner_sigma);
           //C = conv_fft(M,kernel);
           convoluted(i,j) = C(i,j);
          }
        else{
          //kernel1=getGaussian(k_height, k_width, outer_sigma);
          //D = conv_fft(M,kernel1);
          convoluted(i,j) = D(i,j);
        }
      }
  }
  return convoluted;
}




