#include <bits/stdc++.h>
#include <cuComplex.h>
using namespace std;

namespace CPU {
struct Com {
    double a, b;
    Com operator+(const Com x) {
        return (Com){a + x.a, b + x.b};
    }
    Com operator-(const Com x) {
        return (Com){a - x.a, b - x.b};
    }
    Com operator*(const Com x) {
        return (Com){a * x.a - b * x.b, a * x.b + x.a * b};
    }
};
void FFT(double *A, double *res, int n, int *rev) {
    Com *a = new Com[n];
    for (int i = 0; i < n; i++) a[i] = (Com){A[i], 0};
    for (int i = 0; i < n; i++)
        if (rev[i] > i) swap(a[i], a[rev[i]]);
    for (int l = 1; l < n; l <<= 1) {
        Com w0 = (Com){cos(M_PI / l), sin(M_PI / l)};
        for (int i = 0; i < n; i += l << 1) {
            Com w = (Com){1, 0};
            for (int j = i; j < i + l; j++) {
                Com x = a[j], y = a[j + l] * w;
                a[j] = x + y, a[j + l] = x - y, w = w * w0;
            }
        }
    }
    for (int i = 0; i < n; i++) res[i] = a[i].a;
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
__global__ void fft(cuDoubleComplex *a, int l, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = id; i < n; i += blockDim.x * gridDim.x) {
        if (i & l) continue;
        int j = i & (l - 1);
        cuDoubleComplex w =
            make_cuDoubleComplex(cos(j * M_PI / l), sin(j * M_PI / l));
        cuDoubleComplex x = a[i];
        cuDoubleComplex y = cuCmul(a[i + l], w);

        a[i] = cuCadd(x, y);
        a[i + l] = cuCsub(x, y);
    }
}
} // namespace GPU
int main() {
    const int N = 1 << 24;
    const int BLOCK_DIM = 4;
    const int GRID_DIM = 32;
    double *a_cpu = new double[N];
    int *rev = new int[N];
    double *res_cpu = new double[N];
    double *res_gpu = new double[N];
    for (int i = 0; i < N; i++) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << 19);
    for (int i = 0; i < N; i++) a_cpu[i] = i;

    double t0 = clock();
    CPU::FFT(a_cpu, res_cpu, N, rev);

    double t1 = clock();
    // output runtime(s)
    cout << "CPU time: " << (t1 - t0) / CLOCKS_PER_SEC << "s" << endl;

    cuDoubleComplex *a;
    double *a_gpu;
    int *rev_gpu;
    cudaMalloc(&a_gpu, N * sizeof(double));
    cudaMalloc(&rev_gpu, N * sizeof(int));
    cudaMalloc(&a, N * sizeof(cuDoubleComplex));

    cudaMemcpy(a_gpu, a_cpu, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(rev_gpu, rev, N * sizeof(int), cudaMemcpyHostToDevice);

    double t2 = clock();
    cout << "CPU copy to GPU time: " << (t2 - t1) / CLOCKS_PER_SEC << "s"
         << endl;

    GPU::copyToReal<<<GRID_DIM, BLOCK_DIM>>>(a_gpu, a, N);
    // cudaDeviceSynchronize();

    GPU::rev<<<GRID_DIM, BLOCK_DIM>>>(a, rev_gpu, N);
    // cudaDeviceSynchronize();

    for (int l = 1; l < N; l <<= 1) GPU::fft<<<GRID_DIM, BLOCK_DIM>>>(a, l, N);
    // cudaDeviceSynchronize();
    GPU::copyToDouble<<<GRID_DIM, BLOCK_DIM>>>(a_gpu, a, N);

    double t3 = clock();
    cout << "GPU time: " << (t3 - t2) / CLOCKS_PER_SEC << "s" << endl;
    cudaMemcpy(res_gpu, a_gpu, N * sizeof(double), cudaMemcpyDeviceToHost);
    double t4 = clock();
    cout << "GPU copy to CPU time: " << (t4 - t3) / CLOCKS_PER_SEC << "s"
         << endl;
    // cudaDeviceSynchronize();
    for (int i = 0; i < N; i++)
        assert(abs((res_cpu[i] - res_gpu[i]) / res_cpu[i]) < 1e-3);

    return 0;
}
