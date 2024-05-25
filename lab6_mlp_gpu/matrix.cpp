#include "matrix.h"
#include <random>

double &Mat::operator()(int i, int j) const {
    return a[i * m + j];
}
std::mt19937 rng(std::chrono::steady_clock::now()
                     .time_since_epoch()
                     .count());
void Mat::random_init(int N, int M, double loc,
                      double scale) {
    std::normal_distribution<double> dist(loc, scale);
    n = N, m = M;
    a = new double[n * m];
    for (int i = 0; i < n * m; i++) a[i] = dist(rng);
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

Mat Mat::operator*(const Mat &_) const {
    assert(m == _.n);
    Mat res;
    res.zero_init(n, _.m);
    for (int k = 0; k < m; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < _.m; j++)
                res(i, j) += (*this)(i, k) * _(k, j);
    return res;
}
Mat Mat::operator*(const double &_) const {
    Mat res;
    res.zero_init(n, m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            res(i, j) = (*this)(i, j) * _;
    return res;
}

Mat Mat::operator+(const Mat &_) const {
    Mat res;
    res.zero_init(n, m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            res(i, j) = (*this)(i, j) + _(i, j);
    return res;
}
Mat Mat::operator-(const Mat &_) const {
    Mat res;
    res.zero_init(n, m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            res(i, j) = (*this)(i, j) - _(i, j);
    return res;
}

Mat Mat::relu() const {
    Mat res;
    res.zero_init(n, m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            res(i, j) =
                (*this)(i, j) > 0 ? (*this)(i, j) : 0;
    return res;
}

Mat Mat::relu_() const {
    Mat res;
    res.zero_init(n, m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            res(i, j) = (*this)(i, j) > 0 ? 1 : 0;
    return res;
}

Mat Mat::T() const {
    Mat res;
    res.zero_init(m, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            res(j, i) = (*this)(i, j);
    return res;
}

Mat Mat::softmax() const {
    Mat res;
    res.zero_init(n, m);
    assert(n == 1);
    double Max = a[0], sum = 0;
    for (int i = 0; i < m; i++)
        Max = std::max(Max, (*this)(0, i));
    for (int i = 0; i < m; i++)
        sum += exp((*this)(0, i) - Max);

    for (int i = 0; i < m; i++) {
        res(0, i) = exp((*this)(0, i) - Max) / sum;
        // std::cout << (*this)(0, i) << " " << Max << " "
        //           << sum << " " << res(0, i) << " "
        //           << (res(0, i) >= 0) << " "
        //           << (res(0, i) <= 1) << std::endl;
        assert(res(0, i) >= 0 && res(0, i) <= 1);
    }
    return res;
}

Mat Mat::softmax_() const {
    Mat res;
    res.zero_init(n, m);
    assert(n == 1);
    double Max = a[0], sum = 0;
    for (int i = 0; i < m; i++)
        Max = std::max(Max, (*this)(0, i));
    for (int i = 0; i < m; i++)
        sum += exp((*this)(0, i) - Max);
    for (int i = 0; i < m; i++) {
        double x = exp((*this)(0, i) - Max) / sum;
        res(0, i) = x * (1 - x);
        // std::cout << (*this)(0, i) << " " << Max << " "
        //           << sum << " " << res(0, i) << " "
        //           << (res(0, i) >= 0) << " "
        //           << (res(0, i) <= 1) << std::endl;
        assert(res(0, i) >= 0 && res(0, i) <= 1);
    }
    return res;
}

double Mat::sum() const { //根据行求和
    double sum = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) sum += (*this)(i, j);
    return sum;
}

Mat Mat::mult(const Mat &_) const {
    Mat res;
    res.zero_init(n, m);
    // if (m == 10) {
    //     puts("---------");
    //     for (int i = 0; i < m; i++)
    //         printf("%lf ", (*this)(0, i));
    //     puts("");
    //     for (int i = 0; i < m; i++) printf("%lf ", _(0,
    //     i)); puts(""); puts("---------");
    // }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            res(i, j) = (*this)(i, j) * _(i, j);
        }
    return res;
}

double Accuracy(const Mat &a, const Mat &b) {
    // a,b are 1*n matrix
    int Maxa = 0, Maxb = 0;
    // a.print(), b.print();
    for (int i = 0; i < a.m; i++) {
        if (a(0, i) > a(0, Maxa)) Maxa = i;
        if (b(0, i) > b(0, Maxb)) Maxb = i;
    }
    return Maxa == Maxb;
}

double Loss(const Mat &a, const Mat &b) {
    double loss = 0;
    for (int i = 0; i < a.m; i++) {
        loss += -b(0, i) * log(a(0, i));
    }
    return loss;
}