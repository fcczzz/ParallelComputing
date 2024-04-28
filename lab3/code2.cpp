#include <omp.h>
#include <bits/stdc++.h>
using namespace std;
struct Matrix {
    int n, m;
    unsigned int **a;

    Matrix(int n = 0, int m = 0) : n(n), m(m) {
        a = new unsigned int *[n + 1];
        for (int i = 1; i <= n; i++) {
            a[i] = new unsigned int[m + 1];
            memset(a[i], 0, (m + 1) * sizeof(unsigned int));
        }
    }
    Matrix(const Matrix &_) {
        n = _.n, m = _.m;
        a = new unsigned int *[n + 1];
        for (int i = 1; i <= n; i++) {
            a[i] = new unsigned int[m + 1];
            memcpy(a[i], _.a[i],
                   (m + 1) * sizeof(unsigned int));
        }
    }
    ~Matrix() {
        for (int i = 1; i <= n; i++) delete a[i];
        delete a;
    }
    void random_init() {
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++) a[i][j] = rand();
    }
    pair<Matrix, Matrix> split_by_ind(const int &_) const {
        Matrix x(_, m), y(n - _, m);
        for (int i = 1; i <= _; i++)
            memcpy(x.a[i], a[i],
                   (m + 1) * sizeof(unsigned int));
        for (int i = _ + 1; i <= n; i++)
            memcpy(y.a[i - _], a[i],
                   (m + 1) * sizeof(unsigned int));
        return make_pair(x, y);
    }
    pair<Matrix, Matrix> split_by_row(const int &_) const {
        Matrix x(n, _), y(n, m - _);
        for (int i = 1; i <= n; i++) {
#pragma omp parallel for
            for (int j = 1; j <= _; j++)
                x.a[i][j] = a[i][j];
#pragma omp parallel for
            for (int j = _ + 1; j <= m; j++)
                y.a[i][j - _] = a[i][j];
        }
        return make_pair(x, y);
    }
    void copy(const Matrix &_, const int &sx,
              const int &sy) {
#pragma omp parallel for
        for (int i = sx; i < sx + _.n; i++)
            for (int j = sy; j < sy + _.m; j++) {
                a[i][j] = _.a[i - sx + 1][j - sy + 1];
            }
    }
    Matrix operator*(const Matrix &_) const {
        // when m <= 10, stop parallel
        Matrix res(n, _.m);
        assert(m == _.n);
        if (n <= 10 || m <= 10) {
#pragma omp parallel for schedule(dynamic)
            for (int k = 1; k <= m; k++)
                for (int i = 1; i <= n; i++)
                    for (int j = 1; j <= _.m; j++)
                        res.a[i][j] += a[i][k] * _.a[k][j];
        } else {
            int m1 = n >> 1, m2 = _.m >> 1;
            auto sa = this->split_by_ind(m1);
            auto sb = _.split_by_row(m2);

#pragma omp parallel sections
            {
#pragma omp section
                {
                    auto c1 = sa.first * sb.first;
                    res.copy(c1, 1, 1);
                }
#pragma omp section
                {
                    auto c2 = sa.first * sb.second;
                    res.copy(c2, 1, m2 + 1);
                }
#pragma omp section
                {
                    auto c3 = sa.second * sb.first;
                    res.copy(c3, m1 + 1, 1);
                }
#pragma omp section
                {
                    auto c4 = sa.second * sb.second;
                    res.copy(c4, m1 + 1, m2 + 1);
                }
            }
        }
        return res;
    }
};
// const int T = 10; // do T times matrix mult
int main() {
    double t0 = omp_get_wtime();
    unsigned int sum = 0;
    // for (int i = 1; i <= T; i++) {
    for (int i = 1; i <= 1; i++) {
        // int x = rand() % 101 + 400;
        // int y = rand() % 101 + 400;
        // int z = rand() % 101 + 400;
        int x = 2000, y = 2000, z = 2000;
        Matrix a(x, y), b(y, z);
        a.random_init();
        b.random_init();
        Matrix c = a * b;
        sum += c.a[1][1];
    }
    double t1 = omp_get_wtime();
    printf("sum is %u.\nTime elapsed is %lf.\n", sum,
           t1 - t0);
    return 0;
}
