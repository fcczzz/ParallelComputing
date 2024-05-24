#include <bits/stdc++.h>

struct Mat {
    int n, m;
    double *a;
    void random_init(int n, int m);
    void zero_init(int n, int m);
    Mat();
    Mat(Mat &&_);
    Mat(const Mat &_);
    Mat operator=(Mat &&_);
    Mat operator=(const Mat &_);
    ~Mat();
    Mat operator*(const Mat &_);
    Mat operator*(const double &_);
    Mat operator+(const Mat &_);
    Mat operator-(const Mat &_);
    Mat relu();     //激活函数
    Mat relu_();    //激活函数的导数
    Mat softmax();  // softmax函数
    Mat softmax_(); // softmax函数的导数
    Mat T();        //转置
    Mat sum();      //根据行求和
};

double Loss(Mat &y, Mat &y_hat);
double Accuracy(Mat &y, Mat &y_hat);
