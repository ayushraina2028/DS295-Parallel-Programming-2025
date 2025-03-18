#include <bits/stdc++.h>
using namespace std;

#define myInfinity numeric_limits<int>::max()

int main() {
    cout << "myInfinity in C++: " << myInfinity << endl;
    cout << "Maximum Integer in C++: " << numeric_limits<int>::max() << endl;

    if(myInfinity == numeric_limits<int>::max()) {
        cout << "myInfinity is equal to numeric_limits<int>::max() \n";
    }
    else cout << "myInfinity is not equal to numeric_limits<int>::max() \n";

    return 0;
}
