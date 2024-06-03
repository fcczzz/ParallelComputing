#include <bits/stdc++.h>
#include <ctime>
#include <vector>
#include "mnist.h"
#include "matrix.h"

struct MLP {
    Mat W1, W2, b1, b2;
    double lr;

    MLP operator=(const MLP &rhs) {
        W1 = rhs.W1;
        W2 = rhs.W2;
        b1 = rhs.b1;
        b2 = rhs.b2;
        lr = rhs.lr;
        return *this;
    }

    void init(int input_size, int hidden_size, int output_size, double LR) {
        W1.random_init(input_size, hidden_size, 0, sqrt(2.0 / input_size));
        W2.random_init(hidden_size, output_size, 0, sqrt(2.0 / hidden_size));
        b1.random_init(1, hidden_size, 0, sqrt(2.0));
        b2.random_init(1, output_size, 0, sqrt(2.0));

        lr = LR;
    }
    void forward(const Mat &input, Mat &z1, Mat &a1, Mat &z2, Mat &a2) {
        z1 = input * W1 + b1;
        a1 = z1.relu();
        z2 = a1 * W2 + b2;
        a2 = z2.softmax();
    }

    void backward(Mat &input, Mat &z1, Mat &a1, Mat &z2, Mat &a2, Mat &label) {
        Mat delta2 = (a2 - label).mult(z2.softmax_());
        Mat delta1 = (delta2 * W2.T()).mult(z1.relu_());

        Mat dW2 = a1.T() * delta2;
        Mat dW1 = input.T() * delta1;

        W1 = W1 - dW1 * lr;
        W2 = W2 - dW2 * lr;
        b1 = b1 - delta1 * lr;
        b2 = b2 - delta2 * lr;
        // z2.softmax_().print();
        // delta2.print();
        // b2.print();
    }

    std::pair<double, double>
    step(Mat &input, Mat &label,
         bool train = true) { // return loss and accuracy
        Mat z1, a1, z2, a2;
        forward(input, z1, a1, z2, a2);

        double loss = Loss(a2, label);
        double accuracy = Accuracy(a2, label);

        if (train) backward(input, z1, a1, z2, a2, label);
        return std::make_pair(loss, accuracy);
    }
} net, ans;

void train(std::vector<mnist_data> &data, int epoch) {
    net.init(784, 256, 10, 0.01);
    ans = net;
    double best_accuracy = 0;
    double t = clock();
    for (int i = 0; i < epoch; i++) {
        std::vector<double> loss, accuracy;
        int t = 0;
        for (auto &d : data) {
            Mat input, label;
            input.zero_init(1, 784);
            for (int i = 0; i < 784; i++) input.a[i] = d.a[i] / 255.0;

            // input.print();

            label.zero_init(1, 10);
            label.a[d.label] = 1;
            auto res = net.step(input, label);
            loss.push_back(res.first);
            accuracy.push_back(res.second);
            // if (++t % 1000 == 0) {
            //     std::cout
            //         << t << " "
            //         << 1.0 * (clock() - t) / CLOCKS_PER_SEC
            //         << std::endl;
            // }
            // std::cout << res.first << " " << res.second
            //           << std::endl;
            // if (++t == 100) break;
        }
        double average_accuracy =
            std::accumulate(accuracy.begin(), accuracy.end(), 0.0)
            / accuracy.size();
        std::cout << "epoch:" << i << " average_acc:" << average_accuracy
                  << " time used:" << 1.0 * (clock() - t) / CLOCKS_PER_SEC
                  << "s" << std::endl;
        if (average_accuracy > best_accuracy) {
            best_accuracy = average_accuracy;
            ans = net;
        }
    }
}

std::vector<int> test(std::vector<mnist_data> &data) {
    std::vector<int> res;
    for (auto &d : data) {
        Mat input;
        input.zero_init(1, 784);
        for (int i = 0; i < 784; i++) input.a[i] = d.a[i] / 255.0;

        Mat z1, a1, z2, a2;
        ans.forward(input, z1, a1, z2, a2);

        int predict = 0;
        for (int i = 0; i < 10; i++) {
            if (a2.a[i] > a2.a[predict]) predict = i;
        }
        res.push_back(predict);
    }
    return res;
}
int main() {
    std::cout << "start" << std::endl;
    std::vector<mnist_data> train_data = input("./data/train.csv", true);
    std::cout << "train_data.size()" << train_data.size() << std::endl;
    std::vector<mnist_data> test_data = input("./data/test.csv", false);
    train(train_data, 10);
    output("./output/submission.csv", test(test_data));
    return 0;
}