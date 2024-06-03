#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

const int h = 2;

struct Point {
    double x, y, z; // RGB
    Point(double x = 0, double y = 0, double z = 0) :
        x(x), y(y), z(z) {
    }
    Point operator-(const Point &b) {
        return Point(x - b.x, y - b.y, z - b.z);
    }
    Point operator+(const Point &b) {
        return Point(x + b.x, y + b.y, z + b.z);
    }
    Point operator*(double k) {
        return Point(x * k, y * k, z * k);
    }
    Point operator+=(const Point &b) {
        x += b.x;
        y += b.y;
        z += b.z;
        return *this;
    }
};

double dis(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x)
                + (a.y - b.y) * (a.y - b.y)
                + (a.z - b.z) * (a.z - b.z));
}

__device__ double Kernel(double x) {
    //高斯核函数
    // return exp(-x * x / (2 * h * h)) / (sqrt(2 * M_PI) *
    // h);
    return exp(-x / (2 * h * h));
}
__global__ void init_delta(Point p, Point *src,
                           Point *delta, double *w, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        delta[i].x = src[i].x - p.x;
        delta[i].y = src[i].y - p.y;
        delta[i].z = src[i].z - p.z;

        double d = delta[i].x * delta[i].x
                   + delta[i].y * delta[i].y
                   + delta[i].z * delta[i].z;
        w[i] = Kernel(d);
        delta[i].x *= w[i];
        delta[i].y *= w[i];
        delta[i].z *= w[i];
    }
}
__global__ void merge(Point *delta, double *w, int step,
                      int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + step < N && (i & step) == 0
        && (i & (step - 1)) == 0) {
        // merge delta[i] and delta[i + step]
        delta[i].x += delta[i + step].x;
        delta[i].y += delta[i + step].y;
        delta[i].z += delta[i + step].z;
        w[i] += w[i + step];
    }
}
int main() {
    cv::Mat image = cv::imread("input.jpg");
    cv::Mat dst_image = image.clone();

    int n = image.size().height, m = image.size().width;
    int N = n * m;

    int BLOCK_DIM = 1024;
    int GRID_DIM = N / BLOCK_DIM + 1;

    Point *src_array = new Point[N];
    Point *dst_array = new Point[N];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            src_array[i * m + j] =
                Point(pixel[0], pixel[1], pixel[2]);
        }
    }

    Point *src, *delta;
    double *w;
    cudaMalloc(&src, N * sizeof(Point));
    cudaMalloc(&delta, N * sizeof(Point));
    cudaMalloc(&w, N * sizeof(double));
    cudaMemcpy(src, src_array, N * sizeof(Point),
               cudaMemcpyHostToDevice);

    double t_sum = 0;
    for (int i = 0; i < N; i++) {
        Point p = src_array[i];
        if (i % 100 == 0) {
            std::cout << i << " " << i / m << " " << i % m
                      << " " << t_sum / CLOCKS_PER_SEC
                      << std::endl;
        }
        Point nxt = p;
        do {
            p = nxt;
            init_delta<<<GRID_DIM, BLOCK_DIM>>>(
                p, src, delta, w, N);
            cudaDeviceSynchronize();
            double t0 = clock();
            for (int step = 1; step < N; step <<= 1) {
                merge<<<GRID_DIM, BLOCK_DIM>>>(delta, w,
                                               step, N);
            }
            cudaDeviceSynchronize();
            double t1 = clock();
            t_sum += t1 - t0;
            Point sum_delta;
            double sum_w;
            cudaMemcpy(&sum_delta, &delta[0], sizeof(Point),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(&sum_w, &w[0], sizeof(double),
                       cudaMemcpyDeviceToHost);

            nxt = sum_delta * (1 / sum_w) + p;

        } while (dis(p, nxt) > 0.1);
        dst_array[i] = nxt;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            Point p = dst_array[i * m + j];
            int x = p.x + 0.5, y = p.y + 0.5, z = p.z + 0.5;
            x = std::min(255, std::max(0, x));
            y = std::min(255, std::max(0, y));
            z = std::min(255, std::max(0, z));

            dst_image.at<cv::Vec3b>(i, j) =
                cv::Vec3b(p.x, p.y, p.z);
        }
    }
    cv::imwrite("output2.jpg", dst_image);

    delete[] src_array;
    delete[] dst_array;
    cudaFree(src);
    cudaFree(delta);
    cudaFree(w);
    return 0;
}