#include <omp.h>
#include <bits/stdc++.h>
using namespace std;
struct SubMat {
    int sx, sy, ex, ey;
    unsigned int **a;
    SubMat(int sx, int sy, int ex, int ey, unsigned int **a) :
        sx(sx), sy(sy), ex(ex), ey(ey), a(a) {
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

    // split a,b into 4 mats.
    //
    int axm = (a.sx + a.ex) >> 1;
    int aym = (a.sy + a.ey) >> 1;
    int bxm = (b.sx + b.ex) >> 1;
    int bym = (b.sy + b.ey) >> 1;
    SubMat a11(a.sx, a.sy, axm, aym, a.a);
    SubMat a12(a.sx, aym + 1, axm, a.ey, a.a);
    SubMat a21(axm + 1, a.sy, a.ex, aym, a.a);
    SubMat a22(axm + 1, aym + 1, a.ex, a.ey, a.a);

    SubMat b11(b.sx, b.sy, bxm, bym, b.a);
    SubMat b12(b.sx, bym + 1, bxm, b.ey, b.a);
    SubMat b21(bxm + 1, b.sy, b.ex, bym, b.a);
    SubMat b22(bxm + 1, bym + 1, b.ex, b.ey, b.a);

    SubMat c11(a.sx, b.sy, axm, bym, c.a);
    SubMat c12(a.sx, bym + 1, axm, b.ey, c.a);
    SubMat c21(axm + 1, b.sy, a.ex, bym, c.a);
    SubMat c22(axm + 1, bym + 1, a.ex, b.ey, c.a);

#pragma omp parallel sections
    {
#pragma omp section
        Mult(a11, b11, c11);
#pragma omp section
        Mult(a12, b21, c11);
#pragma omp section
        Mult(a11, b12, c12);
#pragma omp section
        Mult(a12, b22, c12);
#pragma omp section
        Mult(a21, b11, c21);
#pragma omp section
        Mult(a22, b21, c21);
#pragma omp section
        Mult(a21, b12, c22);
#pragma omp section
        Mult(a22, b22, c22);
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
            memcpy(a[i], _.a[i], (m + 1) * sizeof(unsigned int));
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
    printf("sum is %u.\nTime elapsed is %lf.\n", sum, t1 - t0);
    return 0;
}
