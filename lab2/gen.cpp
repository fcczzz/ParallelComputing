#include <bits/stdc++.h>
#include <cstdio>
using namespace std;
int n;
int main() {
    srand(time(0));
    cerr << "Input n" << endl;
    scanf("%d", &n);
    printf("%d\n", n);
    for (int i = 1; i <= n; i++) printf("%d ", rand());
    puts("");
    return 0;
}
