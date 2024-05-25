#include <bits/stdc++.h>

struct Mat {
    int n, m;
    double *a;
    double &operator()(int i, int j) const;
    void random_init(int n, int m, double loc,
                     double scale); //随机初始化
    void zero_init(int n, int m);
    Mat();
    Mat(Mat &&_);
    Mat(const Mat &_);
    Mat operator=(Mat &&_);
    Mat operator=(const Mat &_);
    ~Mat();
    Mat operator*(const Mat &_) const;
    Mat operator*(const double &_) const;
    Mat operator+(const Mat &_) const;
    Mat operator-(const Mat &_) const;
    Mat relu() const;     //激活函数
    Mat relu_() const;    //激活函数的导数
    Mat softmax() const;  // softmax函数
    Mat softmax_() const; // softmax函数的导数
    Mat T() const;        //转置
    double sum() const;

    Mat mult(const Mat &_) const; //矩阵对应元素相乘

    void print() const {
        std::cout << "n: " << n << " m: " << m << std::endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
};

double Loss(const Mat &y, const Mat &y_hat);
double Accuracy(const Mat &y, const Mat &y_hat);
