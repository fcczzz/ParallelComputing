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

__global__ void mult_mat_kernel(double *a, double *b,
                                double *c, int n, int m,
                                int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * k) {
        int x = i / k, y = i % k;
        c[i] = 0;
        for (int j = 0; j < m; j++)
            c[i] += a[x * m + j] * b[j * k + y];
    }
}

// double &Mat::operator()(int i, int j) const {
//     return a[i * m + j];
// }
std::mt19937 rng(std::chrono::steady_clock::now()
                     .time_since_epoch()
                     .count());

void Mat::random_init(int N, int M, double loc,
                      double scale) {
    std::normal_distribution<double> dist(loc, scale);
    n = N, m = M;
    cudaMalloc(&a, sizeof(double) * n * m);
    double *host_a = new double[n * m];
    for (int i = 0; i < n * m; i++) host_a[i] = dist(rng);
    cudaMemcpy(a, host_a, sizeof(double) * n * m,
               cudaMemcpyHostToDevice);
    delete[] host_a;
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

Mat::Mat(int n, int m) : n(n), m(m) {
    cudaMalloc(&a, sizeof(double) * n * m);
    cudaMemset(a, 0, sizeof(double) * n * m);
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

void Mult_mat(const Mat &a, const Mat &b, Mat &c) {
    assert(a.m == b.n);
    assert(a.n == c.n);
    assert(b.m == c.m);
    mult_mat_kernel<<<(a.n * b.m + 1023) / 1024, 1024>>>(
        a.a, b.a, c.a, a.n, a.m, b.m);
}
void Mult_num(const Mat &a, double b, Mat &c) {
    assert(a.n == c.n);
    assert(a.m == c.m);
    mult_num_kernel<<<(a.n * a.m + 1023) / 1024, 1024>>>(
        a.a, b, c.a, a.n * a.m);
}

void Add_mat(const Mat &a, const Mat &b, Mat &c) {
    assert(a.n == c.n);
    assert(a.m == c.m);
    add_kernel<<<(a.n * a.m + 1023) / 1024, 1024>>>(
        a.a, b.a, c.a, a.n * a.m);
}

void Sub_mat(const Mat &a, const Mat &b, Mat &c) {
    assert(a.n == c.n);
    assert(a.m == c.m);
    sub_kernel<<<(a.n * a.m + 1023) / 1024, 1024>>>(
        a.a, b.a, c.a, a.n * a.m);
}

void Relu(const Mat &a, Mat &c) {
    assert(a.n == c.n);
    assert(a.m == c.m);
    relu_kernel<<<(a.n * a.m + 1023) / 1024, 1024>>>(
        a.a, c.a, a.n * a.m);
}

void Relu_(const Mat &a, Mat &c) {
    assert(a.n == c.n);
    assert(a.m == c.m);
    relu__kernel<<<(a.n * a.m + 1023) / 1024, 1024>>>(
        a.a, c.a, a.n * a.m);
}

void T(const Mat &a, Mat &c) {
    assert(a.n == c.m);
    assert(a.m == c.n);
    T_kernel<<<(a.n * a.m + 1023) / 1024, 1024>>>(a.a, c.a,
                                                  a.n, a.m);
}

__global__ void softmax_kernel(double *a, double *c,
                               int n) {
    double Max = a[0], sum = 0;
    for (int i = 0; i < n; i++) Max = max(Max, a[i]);
    for (int i = 0; i < n; i++) {
        c[i] = exp(a[i] - Max);
        sum += c[i];
    }
    for (int i = 0; i < n; i++) c[i] /= sum;
}
__global__ void softmax__kernel(double *a, double *c,
                                int n) {
    double Max = a[0], sum = 0;
    for (int i = 0; i < n; i++) Max = max(Max, a[i]);
    for (int i = 0; i < n; i++) {
        c[i] = exp(a[i] - Max);
        sum += c[i];
    }
    for (int i = 0; i < n; i++) {
        c[i] /= sum;
        c[i] = c[i] * (1 - c[i]);
    }
}

void Softmax(const Mat &a, Mat &c) {
    assert(a.n == c.n);
    assert(a.m == c.m);
    softmax_kernel<<<1, 1>>>(a.a, c.a, a.m);
}

void Softmax_(const Mat &a, Mat &c) {
    assert(a.n == c.n);
    assert(a.m == c.m);
    softmax__kernel<<<1, 1>>>(a.a, c.a, a.m);
}
void Mult(const Mat &a, const Mat &b, Mat &c) {
    assert(a.n == c.n);
    assert(a.m == c.m);
    assert(b.n == c.n);
    assert(b.m == c.m);
    mult_kernel<<<(a.n * a.m + 1023) / 1024, 1024>>>(
        a.a, b.a, c.a, a.n * a.m);
}

__global__ void Accuracy_kernel(double *a, double *b, int n,
                                int *c) {
    int Maxa = 0, Maxb = 0;
    for (int i = 0; i < n; i++) {
        if (a[i] > a[Maxa]) Maxa = i;
        if (b[i] > b[Maxb]) Maxb = i;
    }
    *c = Maxa == Maxb;
}

double Accuracy(const Mat &a, const Mat &b) {
    // a,b are 1*n matrix

    static int *c = nullptr;
    if (c == nullptr) cudaMalloc(&c, sizeof(int));
    Accuracy_kernel<<<1, 1>>>(a.a, b.a, a.m, c);
    int host_c;
    cudaMemcpy(&host_c, c, sizeof(int),
               cudaMemcpyDeviceToHost);

    return host_c;
}

__global__ void Loss_kernel(double *a, double *b,
                            double *loss, int n) {
    *loss = 0;
    for (int i = 0; i < n; i++) {
        *loss += a[i] * log(b[i]);
    }
}

double Loss(const Mat &a, const Mat &b) {
    static double *loss = nullptr;
    if (loss == nullptr) cudaMalloc(&loss, sizeof(double));
    Loss_kernel<<<1, 1>>>(a.a, b.a, loss, a.m);
    double host_loss;
    cudaMemcpy(&host_loss, loss, sizeof(double),
               cudaMemcpyDeviceToHost);
    return -host_loss;
}