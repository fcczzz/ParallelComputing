#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>

int main(int argc, char *argv[]) {
    freopen("data.in","r",stdin);
    int nthreads, tid;
    double t0, t1;
    t0 = omp_get_wtime();
    int n;
    scanf("%d",&n);
    int *a = new int[n];

    for(int i=0;i<n;i++)scanf("%d",a+i);
    for(int i=0;i<n;i++){
        for(int j=i&1;j<n-1;j+=2)
            if(a[j]>a[j+1])std::swap(a[j],a[j+1]);
    }

    t1 = omp_get_wtime();
    printf("Time elapsed is %f.\n", t1 - t0);
    return 0;
}
