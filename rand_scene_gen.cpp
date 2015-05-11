#include <iostream>
#include <random>

using namespace std;

#define N 1000
#define LEFT -300
#define RIGHT 300

int main() {
    freopen("scene","w",stdout);
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(LEFT, RIGHT);

    cout << 4 << " " << N << endl;

    for (int i = 0; i < N; i++) {
        int x = distribution(generator); 
        int y = distribution(generator); 
        int z = distribution(generator); 
        
        if (i % 10 == 0) {
			cout << 6 << " " << x << " " << y << " " << z << " " << 100 << " " << 100 << " " << 100 << " " << 0 << " " << 0 << " " << 0 << " " << 1 << endl;
        }
        else {
			cout << 6 << " " << x << " " << y << " " << z << " " << 0 << " " << 0 << " " << 0 << " " << 0.95 << " " << 0.3 << " " << 0.95 << " " << 1 << endl;
        } 
    }

	return 0;
}
