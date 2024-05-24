#include <bits/stdc++.h>

struct Mat {
    int n, m;
    double *a;
    double &operator()(int i, int j) const;
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
    double sum();      

    Mat mult(const Mat &_); //矩阵对应元素相乘
};

double Loss(const Mat &y, const Mat &y_hat);
double Accuracy(const Mat &y, const Mat &y_hat);
