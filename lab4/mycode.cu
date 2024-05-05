#include <bits/stdc++.h>
#include <cuComplex.h>
#include <cufft.h>
using namespace std;

namespace CPU {
void FFT(vector<complex<double>> &a, int n, int *rev, int tp) {
    for (int i = 0; i < n; i++)
        if (rev[i] > i) swap(a[i], a[rev[i]]);
    for (int l = 1; l < n; l <<= 1) {
        complex<double> w0(cos(M_PI / l), tp * sin(M_PI / l));
        for (int i = 0; i < n; i += l << 1) {
            complex<double> w(1, 0);
            for (int j = i; j < i + l; j++) {
                auto x = a[j], y = a[j + l] * w;
                a[j] = x + y, a[j + l] = x - y, w = w * w0;
            }
        }
    }
}
} // namespace CPU
namespace GPU {
__global__ void copyToReal(double *a, cuDoubleComplex *b, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = id; i < n; i += blockDim.x * gridDim.x) {
        b[i] = make_cuDoubleComplex(a[i], 0);
    }
}
__global__ void copyToDouble(double *a, cuDoubleComplex *b, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = id; i < n; i += blockDim.x * gridDim.x) { a[i] = b[i].x; }
}
__global__ void rev(cuDoubleComplex *a, int *rev, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = id; i < n; i += blockDim.x * gridDim.x) {
        if (rev[i] > i) {
            auto t = a[i];
            a[i] = a[rev[i]];
            a[rev[i]] = t;
        }
    }
}
__global__ void init_w(cuDoubleComplex *w, int l, int tp) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = id; i < l; i += blockDim.x * gridDim.x) {
        w[i] = make_cuDoubleComplex(cos(i * M_PI / l), tp * sin(i * M_PI / l));
    }
}
__global__ void fft(cuDoubleComplex *a, int l, int t, int n,
                    cuDoubleComplex *w) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    for (int I = id; I < (n >> 1); I += blockDim.x * gridDim.x) {
        int j = I & (l - 1);
        int i = (I >> t << (t + 1)) | j;
        cuDoubleComplex x = a[i];
        cuDoubleComplex y = cuCmul(a[i + l], w[j]);

        a[i] = cuCadd(x, y);
        a[i + l] = cuCsub(x, y);
    }
}
} // namespace GPU
int main() {
    const int N = 1 << 20;
    const int BLOCK_DIM = 32;
    const int GRID_DIM = 1024;
    double *a_cpu = new double[N];
    double *res_cpu = new double[N];
    double *res_gpu = new double[N];
    vector<complex<double>> a_com(N);
    int *rev = new int[N];
    rev[0] = 0;
    for (int i = 1; i < N; i++)
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) ? N >> 1 : 0);

    for (int i = 0; i < N; i++) a_com[i] = complex<double>(a_cpu[i] = i, 0);

    double t0 = clock();
    CPU::FFT(a_com, N, rev, 1);
    CPU::FFT(a_com, N, rev, -1);

    for (int i = 0; i < N; i++) res_cpu[i] = a_com[i].real();

    double t1 = clock();
    // output runtime(s)
    cout << "CPU time: " << (t1 - t0) / CLOCKS_PER_SEC << "s" << endl;


    cuDoubleComplex *a, *w;
    double *a_gpu;
    int *rev_gpu;
    cudaMalloc(&a_gpu, N * sizeof(double));
    cudaMalloc(&rev_gpu, N * sizeof(int));
    cudaMalloc(&a, N * sizeof(cuDoubleComplex));
    cudaMalloc(&w, N * sizeof(cuDoubleComplex));

    cudaMemcpy(a_gpu, a_cpu, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(rev_gpu, rev, N * sizeof(int), cudaMemcpyHostToDevice);

    double t2 = clock();
    cout << "CPU copy to GPU time: " << (t2 - t1) / CLOCKS_PER_SEC << "s"
         << endl;

    GPU::copyToReal<<<GRID_DIM, BLOCK_DIM>>>(a_gpu, a, N);

    GPU::rev<<<GRID_DIM, BLOCK_DIM>>>(a, rev_gpu, N);
    for (int l = 1, t = 0; l < N; l <<= 1, ++t) {
        GPU::init_w<<<GRID_DIM, BLOCK_DIM>>>(w, l, 1);
        GPU::fft<<<GRID_DIM, BLOCK_DIM>>>(a, l, t, N, w);
    }

    GPU::rev<<<GRID_DIM, BLOCK_DIM>>>(a, rev_gpu, N);
    for (int l = 1, t = 0; l < N; l <<= 1, ++t) {
        GPU::init_w<<<GRID_DIM, BLOCK_DIM>>>(w, l, -1);
        GPU::fft<<<GRID_DIM, BLOCK_DIM>>>(a, l, t, N, w);
    }

    GPU::copyToDouble<<<GRID_DIM, BLOCK_DIM>>>(a_gpu, a, N);
    cudaDeviceSynchronize();

    double t3 = clock();
    cout << "GPU time: " << (t3 - t2) / CLOCKS_PER_SEC << "s" << endl;

    cudaMemcpy(res_gpu, a_gpu, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    double t4 = clock();
    cout << "GPU copy to CPU time: " << (t4 - t3) / CLOCKS_PER_SEC << "s"
         << endl;

    for (int i = 0; i < N; i++) {
        assert(abs(res_cpu[i] - res_gpu[i]) < 1e-3
               || abs((res_cpu[i] - res_gpu[i]) / res_cpu[i]) < 1e-3);
    }

    cudaFree(a);
    cudaFree(w);
    cudaFree(a_gpu);
    cudaFree(rev_gpu);
    return 0;
}
