#include "matrix.h"
#include <random>

__global__ void add_kernel(double *a, double *b, double *c,
                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void sub_kernel(double *a, double *b, double *c,
                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] - b[i];
}

__global__ void relu_kernel(double *a, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] > 0 ? a[i] : 0;
}

__global__ void relu__kernel(double *a, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] > 0 ? 1 : 0;
}

__global__ void softmax_kernel(double *a, double *c, int n,
                               double Max, double sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = exp(a[i] - Max) / sum;
}

__global__ void softmax__kernel(double *a, double *c, int n,
                                double Max, double sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double x = exp(a[i] - Max) / sum;
        c[i] = x * (1 - x);
    }
}

__global__ void T_kernel(double *a, double *c, int n,
                         int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * m) c[i] = a[i % n * m + i / n];
}

__global__ void mult_kernel(double *a, double *b, double *c,
                            int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

__global__ void mult_num_kernel(double *a, double b,
                                double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b;
}

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
    cudaMalloc(&a, sizeof(double) * n * m);
    for (int i = 0; i < n * m; i++) a[i] = dist(rng);
}

void Mat::zero_init(int N, int M) {
    n = N, m = M;
    cudaMalloc(&a, sizeof(double) * n * m);
    cudaMemset(a, 0, sizeof(double) * n * m);
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
    if (a) cudaFree(a);
    n = _.n, m = _.m, a = _.a;
    _.a = nullptr;
    return *this;
}

Mat::Mat(const Mat &_) {
    n = _.n, m = _.m;
    cudaMalloc(&a, sizeof(double) * n * m);
    cudaMemcpy(a, _.a, sizeof(double) * n * m,
               cudaMemcpyDeviceToDevice);
}

Mat Mat::operator=(const Mat &_) {
    if (a) cudaFree(a);
    n = _.n, m = _.m;
    cudaMalloc(&a, sizeof(double) * n * m);
    cudaMemcpy(a, _.a, sizeof(double) * n * m,
               cudaMemcpyDeviceToDevice);
    return *this;
}

Mat::~Mat() {
    if (a) cudaFree(a);
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
    mult_num_kernel<<<(n * m + 255) / 256, 256>>>(
        a, _, res.a, n * m);
    return res;
}

Mat Mat::operator+(const Mat &_) const {
    Mat res;
    res.zero_init(n, m);
    add_kernel<<<(n * m + 255) / 256, 256>>>(a, _.a, res.a,
                                             n * m);
    return res;
}
Mat Mat::operator-(const Mat &_) const {
    Mat res;
    res.zero_init(n, m);
    sub_kernel<<<(n * m + 255) / 256, 256>>>(a, _.a, res.a,
                                             n * m);
    return res;
}

Mat Mat::relu() const {
    Mat res;
    res.zero_init(n, m);
    relu_kernel<<<(n * m + 255) / 256, 256>>>(a, res.a,
                                              n * m);
    return res;
}

Mat Mat::relu_() const {
    Mat res;
    res.zero_init(n, m);
    relu__kernel<<<(n * m + 255) / 256, 256>>>(a, res.a,
                                               n * m);
    return res;
}

Mat Mat::T() const {
    Mat res;
    res.zero_init(m, n);
    T_kernel<<<(n * m + 255) / 256, 256>>>(a, res.a, n, m);
    return res;
}

Mat Mat::softmax() const {
    Mat res;
    res.zero_init(n, m);
    double Max = a[0], sum = 0;
    for (int i = 0; i < m; i++)
        Max = std::max(Max, (*this)(0, i));
    for (int i = 0; i < m; i++)
        sum += exp((*this)(0, i) - Max);

    softmax_kernel<<<(n * m + 255) / 256, 256>>>(
        a, res.a, n * m, Max, sum);
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
    softmax__kernel<<<(n * m + 255) / 256, 256>>>(
        a, res.a, n * m, Max, sum);
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
    mult_kernel<<<(n * m + 255) / 256, 256>>>(a, _.a, res.a,
                                              n * m);
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