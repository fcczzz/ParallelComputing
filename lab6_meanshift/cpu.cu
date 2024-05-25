#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <ostream>

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

double Kernel(double x, double h) {
    //高斯核函数
    return exp(-x * x / (2 * h * h)) / (sqrt(2 * M_PI) * h);
}

int main() {
    cv::Mat image = cv::imread("input.jpg");
    cv::Mat dst_image = image.clone();

    int n = image.size().height, m = image.size().width;
    int N = n * m;
    Point *src_array = new Point[N];
    Point *dst_array = new Point[N];

    int h = 2;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            src_array[i * m + j] =
                Point(pixel[0], pixel[1], pixel[2]);
        }
    }

    std::cout << N << std::endl;
    for (int i = 0; i < N; i++) {
        Point p = src_array[i];
        Point nxt = p;
        // if (i == 18856)
        if (i % 100 == 0)
            std::cout << i << " " << i / m << " " << i % m
                      << " " << p.x << " " << p.y << " "
                      << p.z << std::endl;

        do {
            p = nxt;
            nxt = Point(0, 0, 0);
            double sum = 0;
            for (int j = 0; j < N; j++) {
                Point delta = src_array[j] - p;

                double d = dis(p, src_array[j]);
                double w = Kernel(d, h);

                sum += w;
                nxt += delta * w;
            }
            nxt = nxt * (1 / sum) + p;
        } while (dis(p, nxt) > 0.1);

        dst_array[i] = nxt;
        if (i % 100 == 0)
            std::cout << i << " " << i / m << " " << i % m
                      << " " << p.x << " " << p.y << " "
                      << p.z << std::endl;
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

    cv::imwrite("output.jpg", dst_image);

    delete[] src_array;
    delete[] dst_array;
}