#include <stdlib.h> 
#include <stdio.h>
#include <ctime>
#include <iostream>

#include "gpu.h"

inline double clamp(double x){ return x<0 ? 0 : x>1 ? 1 : x; }

inline int toInt(double x){ return int(pow(clamp(x),1/2.2)*255+.5); }

int main(){
    freopen("../scene", "r", stdin);
    int w=1024, h=768;
    int samps;
    std::cin >> samps;
    samps /= 4;
    Ray cam(Vec(50,52,295.6), Vec(0,-0.042612,-1).norm());
    Vec cx=Vec(w*.5135/h), cy=(cx%cam.d).norm()*.5135, r, *c=new Vec[w*h];
    clock_t t1, t2;
    float mseconds;
     
    t1 = clock();
    ParallelBVH *B = new ParallelBVH();
    t2 = clock();
    mseconds = ((float)(t2 - t1))/ CLOCKS_PER_SEC;
    std::cout << "Initialization took " << mseconds << "\n";
    t1 = clock();
    B->constructRadixTree();
    t2 = clock();
    mseconds = ((float)(t2 - t1))/ CLOCKS_PER_SEC;
    std::cout << "Construct Radix Tree took " << mseconds << "\n";
    t1 = clock();
    B->constructBVHTree();
    t2 = clock();
    mseconds = ((float)(t2 - t1))/ CLOCKS_PER_SEC;
    std::cout << "Construct BVH Tree took " << mseconds << "\n";
    t1 = clock();
    B->optimize();
    t2 = clock();
    mseconds = ((float)(t2 - t1))/ CLOCKS_PER_SEC;
    std::cout << "Optimize BVH Tree took " << mseconds << "\n";
    B->raytrace(c, w, h, samps, cam, cx, cy);

    t1 = clock();
    FILE *f = fopen("image.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i=(h-1); i>=0; i--) {
        for (int j=0; j<w; j++) {
            fprintf(f,"%d %d %d ", toInt(c[i*w+j].x), toInt(c[i*w+j].y), toInt(c[i*w+j].z));
        }
    }
    t2 = clock();
    mseconds = ((float)(t2 - t1))/ CLOCKS_PER_SEC;
    std::cout << "Write to file took " << mseconds << "\n";
}
