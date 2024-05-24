#include <bits/stdc++.h>
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

    void init(int input_size, int hidden_size,
              int output_size, double LR) {
        W1.random_init(hidden_size, input_size);
        W2.random_init(output_size, hidden_size);
        b1.random_init(hidden_size, 1);
        b2.random_init(output_size, 1);

        lr = LR;
    }
    void forward(Mat &input, Mat &z1, Mat &a1, Mat &z2,
                 Mat &a2) {
        z1 = W1 * input + b1;
        a1 = z1.relu();
        z2 = W2 * a1 + b2;
        a2 = z2.softmax();
    }

    void backward(Mat &input, Mat &z1, Mat &a1, Mat &z2,
                  Mat &a2, Mat &label) {
        Mat delta1 = (a2 - label) * z2.softmax_();
        Mat delta2 = W2.T() * delta1 * z1.relu_();

        Mat dW2 = delta1 * a1.T();
        Mat dW1 = delta2 * input.T();
        Mat db2 = delta1.sum();
        Mat db1 = delta2.sum();

        W1 = W1 - dW1 * lr;
        W2 = W2 - dW2 * lr;
        b1 = b1 - db1 * lr;
        b2 = b2 - db2 * lr;
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
    net.init(784, 128, 10, 0.01);
    double best_accuracy = 0;
    for (int i = 0; i < epoch; i++) {
        std::vector<double> loss, accuracy;
        for (auto &d : data) {
            Mat input, label;
            input.zero_init(784, 1);
            memcpy(input.a, d.a, sizeof(d.a));
            label.zero_init(10, 1);
            label.a[d.label] = 1;
            auto res = net.step(input, label);
            loss.push_back(res.first);
            accuracy.push_back(res.second);
        }
        double average_accuracy =
            std::accumulate(accuracy.begin(),
                            accuracy.end(), 0.0)
            / accuracy.size();
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
        input.zero_init(784, 1);
        memcpy(input.a, d.a, sizeof(d.a));

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
    std::vector<mnist_data> train_data =
        input("./data/train.csv", true);
    std::vector<mnist_data> test_data =
        input("./data/test.csv", false);
    train(train_data, 50);
    output("./output/submission.csv", test(test_data));
    return 0;
}