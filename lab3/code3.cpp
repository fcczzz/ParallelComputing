#include <omp.h>
#include <bits/stdc++.h>
using namespace std;
struct SubMat {
    int sx, sy, ex, ey;
    unsigned int **a;
    SubMat(int sx, int sy, int ex, int ey,
           unsigned int **a) :
        sx(sx),
        sy(sy), ex(ex), ey(ey), a(a) {
    }
};

void Mult(SubMat a, SubMat b, SubMat c) {
    // a * b into c;
    if (a.ex - a.sx + 1 <= 10) {
        for (int k = a.sy; k <= a.ey; k++)
            for (int i = a.sx; i <= a.ex; i++) {
                int x = i - a.sx + c.sx;
                for (int j = b.sy; j <= b.ey; j++) {
                    int y = j - b.sy + c.sy;
                    c.a[x][y] += a.a[i][k] * b.a[k][j];
                }
            }
        return;
    }

    // split a,b into 2 mats.

    int axm = (a.sx + a.ex) >> 1;
    int bym = (b.sy + b.ey) >> 1;
    SubMat a1(a.sx, a.sy, axm, a.ey, a.a);
    SubMat a2(axm + 1, a.sy, a.ex, a.ey, a.a);

    SubMat b1(b.sx, b.sy, b.ex, bym, b.a);
    SubMat b2(b.sx, bym + 1, b.ex, b.ey, b.a);

    SubMat c11(c.sx, c.sy, c.sx + axm - a.sx,
               c.sy + bym - b.sy, c.a);
    SubMat c12(c.sx, c.sy + bym - b.sy + 1,
               c.sx + axm - a.sx, c.ey, c.a);
    SubMat c21(c.sx + axm - a.sx + 1, c.sy, c.ex,
               c.sy + bym - b.sy, c.a);
    SubMat c22(c.sx + axm - a.sx + 1, c.sy + bym - b.sy + 1,
               c.ex, c.ey, c.a);

#pragma omp parallel sections
    {
#pragma omp section
        Mult(a1, b1, c11);
#pragma omp section
        Mult(a1, b2, c12);
#pragma omp section
        Mult(a2, b1, c21);
#pragma omp section
        Mult(a2, b2, c22);
    }
}

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
    Matrix operator*(const Matrix &_) const {
        assert(m == _.n);
        Matrix c(n, _.m);

        SubMat x(1, 1, n, m, a);
        SubMat y(1, 1, _.n, _.m, _.a);
        SubMat z(1, 1, n, _.m, c.a);

        Mult(x, y, z);

        return c;
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
