#include <bits/stdc++.h>
// #include <opencv2/opencv.hpp>

using namespace std;

//分别使用cpu和gpu编写锐化卷积核
//-1 -1 -1
//-1 9 -1
//-1 -1 -1
struct Mat {
    int *a, n, m;
    Mat() {
        n = m = 0;
        a = nullptr;
    }
    Mat(int n, int m) : n(n), m(m) {
        a = new int[n * m];
        memset(a, 0, n * m * sizeof(int));
    }
    // move construct
    Mat(Mat &&_) {
        n = _.n, m = _.m;
        a = _.a;
        _.a = nullptr;
    }
    Mat(const Mat &_) {
        n = _.n, m = _.m;
        a = new int[n * m];
        memcpy(a, _.a, n * m * sizeof(int));
    }
    ~Mat() {
        delete[] a;
    }
    int &operator()(int x, int y) {
        return a[x * m + y];
    }
};

namespace CPU {

Mat sharpen(Mat &src) {
    Mat dst(src.n, src.m);
    for (int i = 1; i < src.n - 1; i++) {
        for (int j = 1; j < src.m - 1; j++) {
            int sum = 0;
            sum += src(i - 1, j - 1) * -1;
            sum += src(i - 1, j) * -1;
            sum += src(i - 1, j + 1) * -1;
            sum += src(i, j - 1) * -1;
            sum += src(i, j) * 9;
            sum += src(i, j + 1) * -1;
            sum += src(i + 1, j - 1) * -1;
            sum += src(i + 1, j) * -1;
            sum += src(i + 1, j + 1) * -1;
            dst(i, j) = sum;
        }
    }
    return dst;
}
} // namespace CPU

namespace GPU {
__global__ void sharpen(int *src, int *dest, int n, int m) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = id; i < n * m; i += blockDim.x * gridDim.x) {
        int x = i / m;
        int y = i % m;
        if (x > 0 && x < n - 1 && y > 0 && y < m - 1) {
            int sum = 0;
            sum += src[(x - 1) * m + y - 1] * -1;
            sum += src[(x - 1) * m + y] * -1;
            sum += src[(x - 1) * m + y + 1] * -1;
            sum += src[x * m + y - 1] * -1;
            sum += src[x * m + y] * 9;
            sum += src[x * m + y + 1] * -1;
            sum += src[(x + 1) * m + y - 1] * -1;
            sum += src[(x + 1) * m + y] * -1;
            sum += src[(x + 1) * m + y + 1] * -1;
            dest[x * m + y] = sum;
        }
    }
}
} // namespace GPU

int main() {
    // gen a random mat
    srand(time(0));
    const int N = 1 << 12;
    const int BLOCK_DIM = 4;
    const int GRID_DIM = 32;
    Mat src(N, N);
    for (int i = 0; i < 4096; i++) {
        for (int j = 0; j < 4096; j++) { src(i, j) = rand() % 256; }
    }
    double t0 = clock();
    Mat res = CPU::sharpen(src);
    double t1 = clock();
    cout << "CPU time: " << (t1 - t0) / CLOCKS_PER_SEC << "s" << endl;

    int *gpu_src, *gpu_dst, *gpu_res = new int[N * N];
    cudaMalloc(&gpu_src, N * N * sizeof(int));
    cudaMalloc(&gpu_dst, N * N * sizeof(int));
    cudaMemcpy(gpu_src, src.a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    double t2 = clock();
    cout << "CPU copy to GPU time: " << (t2 - t1) / CLOCKS_PER_SEC << "s"
         << endl;

    GPU::sharpen<<<GRID_DIM, BLOCK_DIM>>>(gpu_src, gpu_dst, N, N);

    cudaDeviceSynchronize();

    double t3 = clock();
    cout << "GPU time: " << (t3 - t2) / CLOCKS_PER_SEC << "s" << endl;

    cudaMemcpy(gpu_res, gpu_dst, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    double t4 = clock();
    cout << "GPU copy to CPU time: " << (t4 - t3) / CLOCKS_PER_SEC << "s"
         << endl;

    // check the result
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) { assert(res(i, j) == gpu_res[i * N + j]); }
    }

    return 0;
}