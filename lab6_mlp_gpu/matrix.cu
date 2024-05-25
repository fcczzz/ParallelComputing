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
    static double sum_t = 0;
    static int cnt = 0;
    cudaDeviceSynchronize();
    double t0 = clock();

    mult_mat_kernel<<<(n * _.m + 1023) / 1024, 1024>>>(
        a, _.a, res.a, n, m, _.m);

    cudaDeviceSynchronize();
    double t1 = clock();
    sum_t += t1 - t0;
    if (++cnt % 1000 == 0)
        std::cout << 1.0 * sum_t / CLOCKS_PER_SEC
                  << std::endl;
    return res;
}
Mat Mat::operator*(const double &_) const {
    Mat res;
    res.zero_init(n, m);
    mult_num_kernel<<<(n * m + 1023) / 1024, 1024>>>(
        a, _, res.a, n * m);
    return res;
}

Mat Mat::operator+(const Mat &_) const {
    Mat res;
    res.zero_init(n, m);
    add_kernel<<<(n * m + 1023) / 1024, 1024>>>(
        a, _.a, res.a, n * m);
    return res;
}
Mat Mat::operator-(const Mat &_) const {
    Mat res;
    res.zero_init(n, m);
    sub_kernel<<<(n * m + 1023) / 1024, 1024>>>(
        a, _.a, res.a, n * m);
    return res;
}

Mat Mat::relu() const {
    Mat res;
    res.zero_init(n, m);
    relu_kernel<<<(n * m + 1023) / 1024, 1024>>>(a, res.a,
                                                 n * m);
    return res;
}

Mat Mat::relu_() const {
    Mat res;
    res.zero_init(n, m);
    relu__kernel<<<(n * m + 1023) / 1024, 1024>>>(a, res.a,
                                                  n * m);
    return res;
}

Mat Mat::T() const {
    Mat res;
    res.zero_init(m, n);
    T_kernel<<<(n * m + 1023) / 1024, 1024>>>(a, res.a, n,
                                              m);
    return res;
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

Mat Mat::softmax() const {
    Mat res;
    res.zero_init(n, m);
    softmax_kernel<<<1, 1>>>(a, res.a, m);
    return res;
}

Mat Mat::softmax_() const {
    static double st = 0;
    static int cnt = 0;
    cudaDeviceSynchronize();
    double t0 = clock();
    Mat res;
    res.zero_init(n, m);

    softmax__kernel<<<1, 1>>>(a, res.a, m);

    cudaDeviceSynchronize();
    double t1 = clock();
    st += t1 - t0;
    if (++cnt % 1000 == 0)
        std::cout << "softmax_:"
                  << 1.0 * st / CLOCKS_PER_SEC << std::endl;

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
    mult_kernel<<<(n * m + 1023) / 1024, 1024>>>(
        a, _.a, res.a, n * m);
    return res;
}

double Accuracy(const Mat &a, const Mat &b) {
    // a,b are 1*n matrix
    int Maxa = 0, Maxb = 0;
    // a.print(), b.print();
    double *host_a = new double[a.m];
    double *host_b = new double[b.m];
    cudaMemcpy(host_a, a.a, sizeof(double) * a.m,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(host_b, b.a, sizeof(double) * b.m,

               cudaMemcpyDeviceToHost);

    for (int i = 0; i < a.m; i++) {
        if (host_a[i] > host_a[Maxa]) Maxa = i;
        if (host_b[i] > host_b[Maxb]) Maxb = i;
    }
    return Maxa == Maxb;
}

double Loss(const Mat &a, const Mat &b) {
    double loss = 0;
    double *host_a = new double[a.m];
    double *host_b = new double[b.m];

    cudaMemcpy(host_a, a.a, sizeof(double) * a.m,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(host_b, b.a, sizeof(double) * b.m,
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < a.m; i++) {
        loss += host_b[i] * log(host_a[i]);
    }
    return loss;
}