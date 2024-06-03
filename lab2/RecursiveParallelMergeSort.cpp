#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <assert.h>
#include <cstring>

int n, *a, *b;

int find(int l, int r, int x) {
    int res = r + 1;
    while (l <= r) {
        int mid = (l + r) >> 1;
        if (a[mid] < x)
            l = mid + 1;
        else
            r = mid - 1, res = mid;
    }
    return res;
}

void Merge(int l1, int r1, int l2, int r2, int lb) {
    // merge a[l1~r1] a[l2~r2] to b[lb~];
    if (r1 - l1 < r2 - l2) {
        std::swap(l1, l2);
        std::swap(r1, r2);
    }
    if (l1 > r1) return;
    if (l2 > r2) {
        memcpy(b + lb, a + l1, sizeof(int) * (r1 - l1 + 1));
        return;
    }

    int mid1 = (l1 + r1) >> 1;
    int mid2 = find(l2, r2, a[mid1]);

    int midb = lb + mid1 - l1 + mid2 - l2;
    b[midb] = a[mid1];
#pragma omp parallel sections
    {
#pragma omp section
        Merge(l1, mid1 - 1, l2, mid2 - 1, lb);
#pragma omp section
        Merge(mid1 + 1, r1, mid2, r2, midb + 1);
    }
}

void Sort(int l, int r) {
    if (l == r) return;
    int mid = (l + r) >> 1;
#pragma omp parallel sections
    {
#pragma omp section
        Sort(l, mid);

#pragma omp section
        Sort(mid + 1, r);
    }
    Merge(l, mid, mid + 1, r, l);
    memcpy(a + l, b + l, sizeof(int) * (r - l + 1));
}

int main(int argc, char *argv[]) {
    freopen("data.in", "r", stdin);
    double t0 = omp_get_wtime();
    scanf("%d", &n);
    a = new int[n + 1];
    b = new int[n + 1];

    for (int i = 1; i <= n; i++) scanf("%d", a + i);

    Sort(1, n);

    for (int i = 1; i < n; i++) assert(a[i] <= a[i + 1]);

    double t1 = omp_get_wtime();
    printf("Time elapsed is %f.\n", t1 - t0);
    return 0;
}
