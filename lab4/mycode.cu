#include <bits/stdc++.h>
#include <cuComplex.h>
#include <cufft.h>
using namespace std;

namespace CPU {
void FFT_recursion(vector<complex<double>> &a, int tp) {
    int n = a.size();

    if (n == 1) return;
    int mid = n >> 1;
    vector<complex<double>> a0, a1;
    for (int i = 0; i < n; i += 2) { //拆分奇偶下标项
        a0.push_back(a[i]);
        a1.push_back(a[i + 1]);
    }
    FFT_recursion(a0, tp);
    FFT_recursion(a1, tp);
    complex<double> w(1, 0);
    complex<double> w0(cos(2 * M_PI / n), tp * sin(2 * M_PI / n));
    for (int i = 0; i < mid; i++, w *= w0) { //合并多项式
        a[i] = a0[i] + w * a1[i];
        a[i + mid] = a0[i] - w * a1[i];
    }
}

// struct Com {
//     double a, b;
//     Com operator+(const Com x) {
//         return (Com){a + x.a, b + x.b};
//     }
//     Com operator-(const Com x) {
//         return (Com){a - x.a, b - x.b};
//     }
//     Com operator*(const Com x) {
//         return (Com){a * x.a - b * x.b, a * x.b + x.a * b};
//     }
// };
// void FFT(double *A, double *res, int n, int *rev) {
//     Com *a = new Com[n];
//     for (int i = 0; i < n; i++) a[i] = (Com){A[i], 0};
//     for (int i = 0; i < n; i++)
//         if (rev[i] > i) swap(a[i], a[rev[i]]);
//     for (int l = 1; l < n; l <<= 1) {
//         Com w0 = (Com){cos(M_PI / l), sin(M_PI / l)};
//         for (int i = 0; i < n; i += l << 1) {
//             Com w = (Com){1, 0};
//             for (int j = i; j < i + l; j++) {
//                 Com x = a[j], y = a[j + l] * w;
//                 a[j] = x + y, a[j + l] = x - y, w = w * w0;
//             }
//         }
//     }
//     for (int i = 0; i < n; i++) res[i] = a[i].a;
// }
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
    const int BLOCK_DIM = 4;
    const int GRID_DIM = 32;
    double *a_cpu = new double[N];
    double *res_cpu = new double[N];
    double *res_gpu = new double[N];
    vector<complex<double>> A(N);
    int *rev = new int[N];
    rev[0] = 0;
    for (int i = 1; i < N; i++)
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) ? N >> 1 : 0);

    for (int i = 0; i < N; i++) a_cpu[i] = i;
    for (int i = 0; i < N; i++) A[i] = complex<double>(i, 0);

    double t0 = clock();
    // CPU::FFT(a_cpu, res_cpu, N, rev);
    CPU::FFT_recursion(A, 1);
    CPU::FFT_recursion(A, -1);

    for (int i = 0; i < N; i++) res_cpu[i] = A[i].real();

    double t1 = clock();
    // output runtime(s)
    cout << "CPU time: " << (t1 - t0) / CLOCKS_PER_SEC << "s" << endl;

    // // using cufft
    // double *a_gpu;
    // cuDoubleComplex *data;
    // cudaMalloc(&a_gpu, N * sizeof(double));
    // cudaMalloc(&data, N * sizeof(cuDoubleComplex));

    // cudaMemcpy(a_gpu, a_cpu, N * sizeof(double), cudaMemcpyHostToDevice);
    // double t2 = clock();
    // GPU::copyToReal<<<GRID_DIM, BLOCK_DIM>>>(a_gpu, data, N);
    // cudaDeviceSynchronize();
    // cufftHandle plan;
    // cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);
    // cufftExecZ2Z(plan, data, data, CUFFT_FORWARD);
    // cufftDestroy(plan);

    // cudaDeviceSynchronize();
    // GPU::copyToDouble<<<GRID_DIM, BLOCK_DIM>>>(a_gpu, data, N);
    // cudaDeviceSynchronize();

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
    cudaDeviceSynchronize();

    GPU::rev<<<GRID_DIM, BLOCK_DIM>>>(a, rev_gpu, N);
    cudaDeviceSynchronize();

    for (int l = 1, t = 0; l < N; l <<= 1, ++t) {
        GPU::init_w<<<GRID_DIM, BLOCK_DIM>>>(w, l, 1);
        cudaDeviceSynchronize();
        GPU::fft<<<GRID_DIM, BLOCK_DIM>>>(a, l, t, N, w);
        cudaDeviceSynchronize();
    }

    GPU::rev<<<GRID_DIM, BLOCK_DIM>>>(a, rev_gpu, N);
    for (int l = 1, t = 0; l < N; l <<= 1, ++t) {
        GPU::init_w<<<GRID_DIM, BLOCK_DIM>>>(w, l, -1);
        cudaDeviceSynchronize();
        GPU::fft<<<GRID_DIM, BLOCK_DIM>>>(a, l, t, N, w);
        cudaDeviceSynchronize();
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

    cudaFree(a);
    cudaFree(w);
    cudaFree(a_gpu);
    cudaFree(rev_gpu);

    for (int i = 0; i < N; i++) {
        assert(abs(res_cpu[i] - res_gpu[i]) < 1e-3
               || abs((res_cpu[i] - res_gpu[i]) / res_cpu[i]) < 1e-3);
    }

    return 0;
}
