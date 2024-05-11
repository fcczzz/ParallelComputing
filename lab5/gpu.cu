#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;

//分别使用cpu和gpu编写锐化卷积核
//-1 -1 -1
//-1 9 -1
//-1 -1 -1

namespace GPU {
__global__ void sharpen(int *src, int *dest, int n, int m) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = id; i < n * m;
         i += blockDim.x * gridDim.x) {
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
    const int BLOCK_DIM = 1024;
    const int GRID_DIM = 1024;

    cv::Mat image = cv::imread("images.jpg");

    std::vector<cv::Mat> image_channels;
    cv::split(image, image_channels);

    double tCpu2Gpu = 0, tGpu = 0, tGpu2Cpu = 0;
    for (int i = 0; i < 3; i++) {
        cv::Mat &src = image_channels[i];
        // cout << src.size().width << " " <<
        // src.size().height << endl; copy src to Mat
        int n = src.size().height, m = src.size().width;
        int *src_array = new int[n * m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++)
                src_array[i * m + j] = src.at<uchar>(i, j);
        }

        int *gpu_src, *gpu_dst, *gpu_res = new int[n * m];
        cudaMalloc(&gpu_src, n * m * sizeof(int));
        cudaMalloc(&gpu_dst, n * m * sizeof(int));
        double t0 = clock();

        cudaMemcpy(gpu_src, src_array, n * m * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        double t1 = clock();

        GPU::sharpen<<<GRID_DIM, BLOCK_DIM>>>(
            gpu_src, gpu_dst, n, m);
        cudaDeviceSynchronize();
        double t2 = clock();

        cudaMemcpy(gpu_res, gpu_dst, n * m * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        double t3 = clock();

        tCpu2Gpu += t1 - t0;
        tGpu += t2 - t1;
        tGpu2Cpu += t3 - t2;

        for (int i = 0; i < src.size().height; i++) {
            for (int j = 0; j < src.size().width; j++) {
                src.at<uchar>(i, j) =
                    max(0, min(255, gpu_res[i * m + j]));
            }
        }
        cudaFree(gpu_src);
        cudaFree(gpu_dst);
        delete[] gpu_res;
        delete[] src_array;
    }

    cout << "CPU copy to GPU time: "
         << tCpu2Gpu / CLOCKS_PER_SEC << "s" << endl;
    cout << "GPU time: " << tGpu / CLOCKS_PER_SEC << "s"
         << endl;
    cout << "GPU copy to CPU time: "
         << tGpu2Cpu / CLOCKS_PER_SEC << "s" << endl;
    cv::merge(image_channels, image);

    cv::namedWindow("image", cv::WINDOW_NORMAL);
    cv::imshow("image", image);
    cv::waitKey(0);

    return 0;
}