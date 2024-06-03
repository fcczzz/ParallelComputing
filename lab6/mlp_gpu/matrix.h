#include <bits/stdc++.h>

struct Mat {
    int n, m;
    double *a;
    void random_init(int n, int m, double loc,
                     double scale); //随机初始化
    void zero_init(int n, int m);
    Mat();
    Mat(int n, int m);
    Mat(Mat &&_);
    Mat(const Mat &_);
    Mat operator=(Mat &&_);
    Mat operator=(const Mat &_);
    ~Mat();
};

void Loss(const Mat &y, const Mat &y_hat,double *loss);
void Accuracy(const Mat &y, const Mat &y_hat,double *accuracy);

// Mat operator*(const Mat &_) const;
// Mat operator*(const double &_) const;
// Mat operator+(const Mat &_) const;
// Mat operator-(const Mat &_) const;
// Mat relu() const;             //激活函数
// Mat relu_() const;            //激活函数的导数
// Mat softmax() const;          // softmax函数
// Mat softmax_() const;         // softmax函数的导数
// Mat T() const;                //转置
// Mat mult(const Mat &_) const; //矩阵对应元素相乘

void Mult_mat(const Mat &a, const Mat &b, Mat &c);
void Mult_num(const Mat &a, double b, Mat &c);
void Add_mat(const Mat &a, const Mat &b, Mat &c);
void Sub_mat(const Mat &a, const Mat &b, Mat &c);
void Relu(const Mat &a, Mat &b);
void Relu_(const Mat &a, Mat &b);
void Softmax(const Mat &a, Mat &b);
void Softmax_(const Mat &a, Mat &b);
void T(const Mat &a, Mat &b);
void Mult(const Mat &a, const Mat &b, Mat &c);
