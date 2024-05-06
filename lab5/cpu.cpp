#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

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

int main() {
    // gen a random mat
    srand(time(0));
    const int N = 1 << 12;
    const int BLOCK_DIM = 1024;
    const int GRID_DIM = 1024;

    // cv::Mat image = cv::imread("images.jpg",
    // cv::IMREAD_GRAYSCALE);
    cv::Mat image = cv::imread("images.jpg");

    std::vector<cv::Mat> image_channels;
    cv::split(image, image_channels);

    double t = 0;
    for (int i = 0; i < 3; i++) {
        cv::Mat &src = image_channels[i];
        // cout << src.size().width << " " <<
        // src.size().height << endl; copy src to Mat
        Mat src_mat(src.size().height, src.size().width);
        for (int i = 0; i < src.size().height; i++) {
            for (int j = 0; j < src.size().width; j++) {
                src_mat(i, j) = src.at<uchar>(i, j);
            }
        }

        double t0 = clock();
        Mat res = CPU::sharpen(src_mat);
        double t1 = clock();
        t += t1 - t0;

        // copy res to dst
        for (int i = 0; i < src.size().height; i++) {
            for (int j = 0; j < src.size().width; j++) {
                src.at<uchar>(i, j) =
                    max(0, min(res(i, j), 255));
            }
        }
    }

    cout << "CPU time: " << t / CLOCKS_PER_SEC << endl;
    cv::merge(image_channels, image);

    cv::namedWindow("image", cv::WINDOW_NORMAL);
    cv::imshow("image", image);
    cv::waitKey(0);
    return 0;
}