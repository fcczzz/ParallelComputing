#include "matrix.h"

double &Mat::operator()(int i, int j) const {
    return a[i * m + j];
}
void Mat::random_init(int N, int M) {
    n = N, m = M;
    a = new double[n * m];
    for (int i = 0; i < n * m; i++) {
        a[i] = 0.5 - (double)rand() / RAND_MAX;
    }
}
void Mat::zero_init(int N, int M) {
    n = N, m = M;
    a = new double[n * m];
    memset(a, 0, sizeof(double) * n * m);
}

Mat::Mat() {
    n = m = 0;
    a = nullptr;
}

Mat::Mat(Mat &&_) {
    n = _.n;
    m = _.m;
    a = _.a;
    _.a = nullptr;
}

Mat Mat::operator=(Mat &&_) {
    if (a) delete[] a;
    n = _.n;
    m = _.m;
    a = _.a;
    _.a = nullptr;
    return *this;
}

Mat::Mat(const Mat &_) {
    n = _.n;
    m = _.m;
    a = new double[n * m];
    memcpy(a, _.a, sizeof(double) * n * m);
}

Mat Mat::operator=(const Mat &_) {
    if (a) delete[] a;
    n = _.n;
    m = _.m;
    a = new double[n * m];
    memcpy(a, _.a, sizeof(double) * n * m);
    return *this;
}

Mat::~Mat() {
    if (a) delete[] a;
}

Mat Mat::operator*(const Mat &_) {
    assert(m == _.n);
    Mat res;
    res.zero_init(n, _.m);
    for (int k = 0; k < m; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < _.m; j++)
                res(i, j) += (*this)(i, k) * _(k, j);
    return res;
}
Mat Mat::operator*(const double &_) {
    Mat res;
    res.zero_init(n, m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            res(i, j) = (*this)(i, j) * _;
    return res;
}

Mat Mat::operator+(const Mat &_) {
    Mat res;
    res.zero_init(n, m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            res(i, j) = (*this)(i, j) + _(i, j);
    return res;
}
Mat Mat::operator-(const Mat &_) {
    Mat res;
    res.zero_init(n, m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            res(i, j) = (*this)(i, j) - _(i, j);
    return res;
}

Mat Mat::relu() {
    Mat res;
    res.zero_init(n, m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            res(i, j) =
                (*this)(i, j) > 0 ? (*this)(i, j) : 0;
    return res;
}

Mat Mat::relu_() {
    Mat res;
    res.zero_init(n, m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            res(i, j) = (*this)(i, j) > 0 ? 1 : 0;
    return res;
}

Mat Mat::T() {
    Mat res;
    res.zero_init(m, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            res(j, i) = (*this)(i, j);
    return res;
}

Mat Mat::softmax() {
    Mat res;
    res.zero_init(n, m);
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < m; j++)
            sum += exp((*this)(i, j));
        for (int j = 0; j < m; j++)
            res(i, j) = exp((*this)(i, j)) / sum;
    }
    return res;
}

Mat Mat::softmax_() {
    Mat res;
    res.zero_init(n, m);
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < m; j++)
            sum += exp((*this)(i, j));
        for (int j = 0; j < m; j++)
            res(i, j) = exp((*this)(i, j)) / sum
                        * (1 - exp((*this)(i, j)) / sum);
    }
    return res;
}

double Mat::sum() { //根据行求和
    double sum = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) sum += (*this)(i, j);
    return sum;
}

Mat Mat::mult(const Mat &_) {
    Mat res;
    res.zero_init(n, m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            res(i, j) = (*this)(i, j) * _(i, j);
    return res;
}

double Accuracy(const Mat &a, const Mat &b) {
    // a,b are n*1 matrix
    int n = a.n;
    int cnt = 0;
    for (int i = 0; i < n; i++)
        if (a(i, 0) == b(i, 0)) cnt++;
    return (double)cnt / n;
}

double Loss(const Mat &a, const Mat &b) {
    // a,b are n*1 matrix
    int n = a.n;
    double loss = 0;
    for (int i = 0; i < n; i++)
        loss -= b(i, 0) * log(a(i, 0));
    return loss;
}