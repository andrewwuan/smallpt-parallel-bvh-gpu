// Parallel BVH Implementation

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <math.h>
#include <stdlib.h> 
#include <stdio.h> 
#include <iostream>
#include <queue>
#include <ctime>

#include "gpu.h"

#define ARRAY_SIZE(array) (sizeof((array))/sizeof((array[0])))
#define BLOCK_SIZE 256
#define CODE_OFFSET (1<<21)
#define CODE_LENGTH (21)
#define INTERSECT_STACK_SIZE (18)
#define RESTRUCT_STACK_SIZE (4)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define Ci 1.2
#define Cl 0.0
#define Ct 1.0

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline __device__ double clamp(double x){ return x<0 ? 0 : x>1 ? 1 : x; }

inline __device__ int toInt(double x){ return int(pow(clamp(x),1/2.2)*255+.5); }

__device__ double drandom(curandState *s) {
    double d = curand_uniform_double(s);
    return d;
}

inline __device__ float min2(float a, float b) {
    return (a < b) ? a : b;
}

inline __device__ float max2(float a, float b) {
    return (a > b) ? a : b;
}

__device__ void merge_bounds(Bound& b1, Bound& b2, Bound* b3) {
   b3->min_x = min2(b1.min_x, b2.min_x);
   b3->max_x = max2(b1.max_x, b2.max_x);
   b3->min_y = min2(b1.min_y, b2.min_y);
   b3->max_y = max2(b1.max_y, b2.max_y);
   b3->min_z = min2(b1.min_z, b2.min_z);
   b3->max_z = max2(b1.max_z, b2.max_z);
   return;
}

inline __device__ int intMin(int i, int j) {
        return (i > j) ? j : i;
}

inline __device__ int intMax(int i, int j) {
        return (i > j) ? i : j;
}

/**
 * Longest common prefix for morton code
 */
inline __device__ int longestCommonPrefix(int i, int j, int len) {
    if (0 <= j && j < len) {
        return __clz(i ^ j);
    } else {
        return -1;
    }
}

/**
 * Test if a ray intersect a bound
 */
__device__ bool intersection_bound_test(const Ray &r, Bound& bound) {
    float t_min, t_max, t_xmin, t_xmax, t_ymin, t_ymax, t_zmin, t_zmax;
    float x_a = 1.0/r.d.x, y_a = 1.0/r.d.y, z_a = 1.0/r.d.z;
    float  x_e = r.o.x, y_e = r.o.y, z_e = r.o.z;

    // calculate t interval in x-axis
    if (x_a >= 0) {
        t_xmin = (bound.min_x - x_e) * x_a;
        t_xmax = (bound.max_x - x_e) * x_a;
    }
    else {
        t_xmin = (bound.max_x - x_e) * x_a;
        t_xmax = (bound.min_x - x_e) * x_a;
    }

    // calculate t interval in y-axis
    if (y_a >= 0) {
        t_ymin = (bound.min_y - y_e) * y_a;
        t_ymax = (bound.max_y - y_e) * y_a;
    }
    else {
        t_ymin = (bound.max_y - y_e) * y_a;
        t_ymax = (bound.min_y - y_e) * y_a;
    }

    // calculate t interval in z-axis
    if (z_a >= 0) {
        t_zmin = (bound.min_z - z_e) * z_a;
        t_zmax = (bound.max_z - z_e) * z_a;
    }
    else {
        t_zmin = (bound.max_z - z_e) * z_a;
        t_zmax = (bound.min_z - z_e) * z_a;
    }

    // find if there an intersection among three t intervals
    t_min = max2(t_xmin, max2(t_ymin, t_zmin));
    t_max = min2(t_xmax, min2(t_ymax, t_zmax));

    return (t_min <= t_max);
}

/**
 * Intersect test in BVH
 */
 __device__ bool intersect(Sphere *start, TreeNode *cur, const Ray &r, 
    double &t, int &id) {

    // int n = 9;
    // double d, inf=t=1e20;
    // for(int i=n;i--;) if((d=start[i].intersect(r))&&d<t){t=d;id=i;}
    // return t<inf;

    // Use static allocation because malloc() can't be called in parallel
    // Use stack to traverse BVH to save space (cost is O(height))
    TreeNode *stack[INTERSECT_STACK_SIZE];
    int topIndex = INTERSECT_STACK_SIZE;
    stack[--topIndex] = cur;
    bool intersected = false;

    // Do while stack is not empty
    while (topIndex != INTERSECT_STACK_SIZE) {
        TreeNode *n = stack[topIndex++];
        if (intersection_bound_test(r, n->bound)) {
            if (n->leaf) {
                double d = n->sphere->intersect(r);
                if (d != 0.0) {
                    if (d < t) {
                        t = d;
                        id = n->sphere->index;
                    }
                    intersected = true;
                }
            } else {
                stack[--topIndex] = n->right;
                stack[--topIndex] = n->left;

                if (topIndex < 0) {
                    printf("Intersect stack not big enough. Increase INTERSECT_STACK_SIZE!\n");
                    return false;
                }
            }
        }
    }

    return intersected;
}

__device__ Vec radiance(Sphere *start, TreeNode *cur, const Ray &r_, int depth_, 
    curandState *s){
  double t;                               // distance to intersection
  int id=0;                               // id of intersected object
  Ray r=r_;
  int depth=depth_;
  Vec cl(0,0,0);   // accumulated color
  Vec cf(1,1,1);  // accumulated reflectance
  while (1){
    t = 1e20;
    if (!intersect(start, cur, r, t, id)) return cl; // if miss, return black
    Sphere &obj = start[id];        // the hit object
    Vec x=r.o+r.d*t, n=(x-obj.p).norm(), nl=n.dot(r.d)<0?n:n*-1, f=obj.c;
    double p = f.x>f.y && f.x>f.z ? f.x : f.y>f.z ? f.y : f.z; // max refl
    cl = cl + cf.mult(obj.e);
    if (++depth>5) if (drandom(s)<p) f=f*(1/p); else return cl; //R.R.
    cf = cf.mult(f);
    if (obj.refl == DIFF){                  // Ideal DIFFUSE reflection
      double r1=2*M_PI*drandom(s), r2=drandom(s), r2s=sqrt(r2);
      Vec w=nl, u=((fabs(w.x)>.1?Vec(0,1):Vec(1))%w).norm(), v=w%u;
      Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).norm();
      r = Ray(x,d);
      continue;
    } else if (obj.refl == SPEC){           // Ideal SPECULAR reflection
      r = Ray(x,r.d-n*2*n.dot(r.d));
      continue;
    }
    Ray reflRay(x, r.d-n*2*n.dot(r.d));     // Ideal dielectric REFRACTION
    bool into = n.dot(nl)>0;                // Ray from outside going in?
    double nc=1, nt=1.5, nnt=into?nc/nt:nt/nc, ddn=r.d.dot(nl), cos2t;
    if ((cos2t=1-nnt*nnt*(1-ddn*ddn))<0){    // Total internal reflection
      r = reflRay;
      continue;
    }
    Vec tdir = (r.d*nnt - n*((into?1:-1)*(ddn*nnt+sqrt(cos2t)))).norm();
    double a=nt-nc, b=nt+nc, R0=a*a/(b*b), c = 1-(into?-ddn:tdir.dot(n));
    double Re=R0+(1-R0)*c*c*c*c*c,Tr=1-Re,P=.25+.5*Re,RP=Re/P,TP=Tr/(1-P);
    if (drandom(s)<P){
      cf = cf*RP;
      r = reflRay;
    } else {
      cf = cf*TP;
      r = Ray(x,tdir);
    }
    continue;
  }
}

/**
 * Ray trace kernel
 * Use BVH for better performance
 */
 __global__ void kernelRayTrace(curandState* states, Vec *deviceSubpixelBuffer, int width, 
    int height, int samps, Sphere *start, TreeNode *cudaDeviceTreeNodes, 
    Ray cam, Vec cx, Vec cy) {

    int subpixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (subpixelIndex >= width * height * 4 || subpixelIndex < 0) return;
    int pixelIndex = subpixelIndex / 4;
    
    int y = pixelIndex / width;
    int x = pixelIndex % width;
    int sy = (subpixelIndex % 4) / 2;
    int sx = (subpixelIndex % 4) % 2;

    if (x < 0 || y < 0 || x >= width || y >= height) {
        return;
    }

    curand_init(y*y*y, subpixelIndex, 0, &states[subpixelIndex]);
    curandState state = states[subpixelIndex];
    
    Vec r = Vec();
    for (int s=0; s<samps; s++) {
        double r1=2*drandom(&state), dx=r1<1 ? sqrt(r1)-1: 1-sqrt(2-r1);
        double r2=2*drandom(&state), dy=r2<1 ? sqrt(r2)-1: 1-sqrt(2-r2);
        Vec d = cx*( ( (sx+.5 + dx)/2 + x)/width - .5) +
                cy*( ( (sy+.5 + dy)/2 + y)/height - .5) + cam.d;
        r = r + radiance(start, cudaDeviceTreeNodes, Ray(cam.o+d*140,d.norm()), 0, &state) * (1./samps);
    }

    deviceSubpixelBuffer[subpixelIndex] = r;
}

/**
 * Get result kernel
 * Combine subpixel colors into one
 */
 __global__ void kernelGetResult(Vec *deviceSubpixelBuffer,
    Vec *devicePixelBuffer, int width, int height) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width * height) {
        return;
    }

    Vec res = Vec();
    for (int i = 0; i < 4; i++) {
        Vec subpixelVec = deviceSubpixelBuffer[index * 4 + i];
        res = res + Vec(clamp(subpixelVec.x), clamp(subpixelVec.y),
        clamp(subpixelVec.z)) * .25;
    }

    devicePixelBuffer[index] = res;
 }

/**
 * Radix tree construction kernel
 * Algorithm described in karras2012 paper.
 * Node-wise parallel
 */
 __global__ void kernelConstructRadixTree(int len, 
    TreeNode *radixTreeNodes, TreeNode *radixTreeLeaves) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= len) return;

    // Run radix tree construction algorithm
    // Determine direction of the range (+1 or -1)
    int d = longestCommonPrefix(i, i + 1, len+1) - 
    longestCommonPrefix(i, i - 1, len+1) > 0 ? 1 : -1;
    
    // Compute upper bound for the length of the range
    int sigMin = longestCommonPrefix(i, i - d, len+1);
    int lmax = 2;

    while (longestCommonPrefix(i, i + lmax * d, len+1) > sigMin) {
        lmax *= 2;
    }

    // Find the other end using binary search
    int l = 0;
    int divider = 2;
    for (int t = lmax / divider; t >= 1; divider *= 2) {
        if (longestCommonPrefix(i, i + (l + t) * d, len+1) > sigMin) {
            l += t;
        }
        t = lmax / divider;
    }
  
    int j = i + l * d;
  
    
    //printf("i:%d d:%d lmax:%d l:%d j:%d \n",i , d, lmax, l, j);
    // Find the split position using binary search
    int sigNode = longestCommonPrefix(i, j, len+1);
    int s = 0;
    divider = 2;
    for (int t = (l + (divider - 1)) / divider; t >= 1; divider *= 2) {
        if (longestCommonPrefix(i, i + (s + t) * d, len+1) > sigNode) {
            s = s + t;
        }
        t = (l + (divider - 1)) / divider;
    }

    int gamma = i + s * d + intMin(d, 0);

    // Output child pointers
    TreeNode *current = radixTreeNodes + i;


    if (intMin(i, j) == gamma) {
        current->left = radixTreeLeaves + gamma;
        (radixTreeLeaves + gamma)->parent = current;
    } else {
        current->left = radixTreeNodes + gamma;
        (radixTreeNodes + gamma)->parent = current;
    }

    if (intMax(i, j) == gamma + 1) {
        current->right = radixTreeLeaves + gamma + 1;
        (radixTreeLeaves + gamma + 1)->parent = current;
    } else {
        current->right = radixTreeNodes + gamma + 1;
        (radixTreeNodes + gamma + 1)->parent = current;
    }

    current->min = intMin(i, j);
    current->max = intMax(i, j);
}

__device__ bool check_bound(TreeNode *p, TreeNode *l, TreeNode *r) {
    return (
        p->bound.min_x == min2(l->bound.min_x, r->bound.min_x) &&
        p->bound.max_x == max2(l->bound.max_x, r->bound.max_x) &&
        p->bound.min_y == min2(l->bound.min_y, r->bound.min_y) &&
        p->bound.max_y == max2(l->bound.max_y, r->bound.max_y) &&
        p->bound.min_z == min2(l->bound.min_z, r->bound.min_z) &&
        p->bound.max_z == max2(l->bound.max_z, r->bound.max_z)
    );
}

__device__ bool check_sanity(TreeNode *n) {
    if (n->leaf) {
        return true;
    } else {
        return (
            n->left->parent == n &&
            n->right->parent == n
        );
    }
}

__device__ void printBVH(TreeNode *root) {

    int level = 1;
    Queue *q = new Queue();
    q->push((void *)root);

    Queue *qt = new Queue();

    while (!q->empty()) {

        printf("\n######### Level %d ##########\n", level++);
        while (!q->empty()) {
            TreeNode *n = (TreeNode *)(q->last());
            q->pop();
            printf("(%d %d) %p", n->min, n->max, n);

            if (!check_sanity(n)) {
                printf(" !SanityError! ");
            }

            if (!n->leaf) {
                if (!check_bound(n, n->left, n->right)) {
                    printf(" !BoundError!");
                }
                printf("\n");
                qt->push((void *)n->left);
                qt->push((void *)n->right);
            } else {
                printf(" ((A:%.0lf C:%.0lf) Sphere: %d)\n", n->area, n->cost, n->sphere->index);
            }
        }
        printf("\n");

        Queue *t = q;
        q = qt;
        qt = t;
    }

    printf("\n");

    delete q;
    delete qt;
}

__global__ void kernelPrintBVH(TreeNode *root) {
    printBVH(root);
}

/**
 * BVH Construction kernel
 * Algorithm described in karras2012 paper (bottom-up approach).
 */
 __global__ void kernelConstructBVHTree(int len, 
    TreeNode *treeNodes, TreeNode *treeLeaves, int *nodeCounter,
    int *sorted_geometry_indices, Sphere *spheres) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= len) return;

    TreeNode *leaf = treeLeaves + index;

    // Handle leaf first
    int geometry_index = sorted_geometry_indices[index];
    leaf->bound = spheres[geometry_index].bound;
    leaf->sphere = &(spheres[geometry_index]);

    TreeNode *current = leaf->parent;
    int currentIndex = current - treeNodes;
    int res = atomicAdd(nodeCounter + currentIndex, 1);

    // Go up and handle internal nodes
    while (1) {
        if (res == 0) {
            return;
        }

        merge_bounds(current->left->bound, current->right->bound,
            &(current->bound));

        // If current is root, return
        if (current == treeNodes) {
            return;
        }
        current = current->parent;
        currentIndex = current - treeNodes;
        res = atomicAdd(nodeCounter + currentIndex, 1);
    }
}

__inline__ __device__ float getArea(float min_x, float max_x,
    float min_y, float max_y, float min_z, float max_z) {
    float dx = max_x - min_x;
    float dy = max_y - min_y;
    float dz = max_z - min_z;
    return 2 * (dx * dy + dx * dz + dy * dz);
}

__device__ int pwr(int base, unsigned exp) {
    int acc = 1;
    for (unsigned c = 0; c < exp; c++) {
        acc *= base;
    }
    return acc;
}

__device__ float get_total_area(int n, TreeNode *leaves[], unsigned s) {
    float lmin_x, lmin_y, lmin_z, lmax_x, lmax_y, lmax_z;
    float min_x = pos_infinity;
    float max_x = neg_infinity;
    float min_y = pos_infinity;
    float max_y = neg_infinity;
    float min_z = pos_infinity;
    float max_z = neg_infinity; 
    for (int i = 0; i < n; i++) {
        if ((s >> i) & 1 == 1) {
            lmin_x = leaves[i]->bound.min_x;
            lmin_y = leaves[i]->bound.min_y;
            lmin_z = leaves[i]->bound.min_z;
            lmax_x = leaves[i]->bound.max_x;
            lmax_y = leaves[i]->bound.max_y;
            lmax_z = leaves[i]->bound.max_z;
            if (lmin_x < min_x) min_x = lmin_x;
            if (lmin_y < min_y) min_y = lmin_y;
            if (lmin_z < min_z) min_z = lmin_z;
            if (lmax_x > max_x) max_x = lmax_x;
            if (lmax_y > max_y) max_y = lmax_y;
            if (lmax_z > max_z) max_z = lmax_z;
        }
    } 
    return getArea(min_x, max_x, min_y, max_y, min_z, max_z);
}

__device__ void calculateOptimalTreelet(int n, TreeNode **leaves, 
                                        unsigned char *p_opt) {
    int num_subsets = pwr(2, n) - 1;
    // 0th element in array should not be used
    float a[128];
    float c_opt[128];
    // Calculate surface area for each subset
    for (unsigned char s = 1; s <= num_subsets; s++) {
        a[s] = get_total_area(n, leaves, s);
    }
    // Initialize costs of individual leaves
    for (unsigned i = 0; i <= (n-1); i++) {
        c_opt[pwr(2, i)] = leaves[i]->cost;
    }
    // Optimize every subset of leaves
    for (unsigned k = 2; k <= n; k++) {
        for (unsigned char s = 1; s <= num_subsets; s++) {
            if (__popc(s) == k) {
                // Try each way of partitioning the leaves
                float c_s = pos_infinity;
                unsigned char p_s = 0;
                unsigned char d = (s - 1) & s;
                unsigned char p = (-d) & s;
                while (p != 0) {
                    float c = c_opt[p] + c_opt[s ^ p];
                    if (c < c_s) {
                        c_s = c;
                        p_s = p;
                    }
                    //printf("p=%x, c=%.0lf, c_s=%.0lf, p_s=%x\n", p & 0xff, c, c_s, p_s & 0xff);
                    p = (p - d) & s;
                }
                // Calculate final SAH cost
                c_opt[s] = Ci * a[s] + c_s;
                p_opt[s] = p_s;
            }
        }
    }
}

__device__ void propagateAreaCost(TreeNode *root, TreeNode **leaves, int num_leaves) {

    for (int i = 0; i < num_leaves; i++) {
        TreeNode *cur = leaves[i];
        cur = cur->parent;
        while (cur != root) {
            if (cur->cost == 0.0) {
                if (cur->left->cost != 0.0 && cur->right->cost != 0.0) {
                    // Both left & right propagated
                    Bound *bound = &cur->bound;
                    merge_bounds(cur->left->bound, cur->right->bound, bound);
                    cur->area = getArea(bound->min_x, bound->max_x, bound->min_y,
                        bound->max_y, bound->min_z, bound->max_z);
                    cur->cost = Ci * cur->area + cur->left->cost + cur->right->cost;
                } else {
                    // Only one side propagated
                    break;
                }
            }
            cur = cur->parent;
        }
    }

    // Propagate root
    Bound *bound = &root->bound;
    merge_bounds(root->left->bound, root->right->bound, bound);
    root->area = getArea(bound->min_x, bound->max_x, bound->min_y,
        bound->max_y, bound->min_z, bound->max_z);
    root->cost = Ci * root->area + root->left->cost + root->right->cost;
}

struct PartitionEntry {
    unsigned char partition;
    bool left;
    TreeNode *parent;
};

__device__ void restructTree(TreeNode *parent, TreeNode **leaves,
    TreeNode **nodes, unsigned char partition, unsigned char *optimal,
    int &index, bool left, int num_leaves) {

    PartitionEntry stack[RESTRUCT_STACK_SIZE];
    int topIndex = RESTRUCT_STACK_SIZE;
    PartitionEntry tmp = {partition, left, parent};
    stack[--topIndex] = tmp;

    // Do while stack is not empty
    while (topIndex != RESTRUCT_STACK_SIZE) {
        PartitionEntry *pe = &stack[topIndex++];
        partition = pe->partition;
        left = pe->left;
        parent = pe->parent;

#ifdef DEBUG_PRINT
        printf("parent=%p partition=%x(%s) index=%d left=%d\n", parent, partition & 0xff, 
            (__popc(partition) == 1 ? "leaf" : "node"), index, left);
#endif

#ifdef DEBUG_CHECK
        if (partition == 0) {
            printf("partition is 0\n", partition);
            return;
        }
#endif

        if (__popc(partition) == 1) {
            // Leaf
            int leaf_index = __ffs(partition) - 1;
#ifdef DEBUG_PRINT
            printf("parent=%p leaf_index=%d\n", parent, leaf_index);
#endif

            TreeNode *leaf = leaves[leaf_index];
            if (left) {
                parent->left = leaf;
            } else {
                parent->right = leaf;
            }
            leaf->parent = parent;
        } else {
            // Internal node
#ifdef DEBUG_CHECK
            if (index >= 7) {
                printf("index out of range\n");
                return;
            }
#endif

            TreeNode *node = nodes[index++];

            // Set cost to 0 as a mark
            node->cost = 0.0;

            if (left) {
                parent->left = node;
            } else {
                parent->right = node;
            }
            node->parent = parent;

#ifdef DEBUG_CHECK
            if (partition >= 128) {
                printf("partition out of range\n");
                return;
            }
#endif
            unsigned char left_partition = optimal[partition];
            unsigned char right_partition = (~left_partition) & partition;

#ifdef DEBUG_CHECK
            if ((left_partition | partition) != partition) {
                printf("left error: %x vs %x\n", left_partition & 0xff, partition & 0xff);
                return;
            }
            if ((right_partition | partition) != partition) {
                printf("right error: %x vs %x\n", right_partition & 0xff, partition & 0xff);
                return;
            }
#endif

#ifdef DEBUG_CHECK
            if (topIndex < 2) {
                printf("restructTree stack not big enough. Increase RESTRUCT_STACK_SIZE!\n");
            }
#endif
            PartitionEntry tmp1 = {left_partition, true, node};
            stack[--topIndex] = tmp1;
            PartitionEntry tmp2 = {right_partition, false, node};
            stack[--topIndex] = tmp2;
        }
    }

    propagateAreaCost(parent, leaves, num_leaves);
}

__device__ void printPartition(TreeNode *root, unsigned char *optimal,
    unsigned char start, unsigned char mask) {
    int level = 1;
    Queue *q = new Queue();
    q->push((void *)start);
    q->push((void *)((~start) & mask));

    Queue *qt = new Queue();

    while (!q->empty()) {

        while (!q->empty()) {
            unsigned char n = (unsigned char)(unsigned long)(q->last());
            q->pop();

            if (__popc(n) != 1) {
                printf("[%d %p] %x\n", level, root, n & 0xff);
                qt->push((void *)optimal[n]);
                qt->push((void *)((~optimal[n]) & n));
            } else {
                printf("[%d %p] (%d)\n", level, root, __ffs(n));
            }
        }
        level++;
        printf("\n");

        Queue *t = q;
        q = qt;
        qt = t;
    }

    printf("\n");

    delete q;
    delete qt;
}

/**
 * treeletOptimize
 * Find the treelet and optimize
 */
__device__ void treeletOptimize(TreeNode *root) {
    // Don't need to optimize if root is a leaf
    if (root->leaf) return;

    // Find a treelet with max number of leaves being 7
    TreeNode *leaves[7];
    int counter = 0;
    leaves[counter++] = root->left;
    leaves[counter++] = root->right;

    // Also remember the internal nodes
    // Max 7 (leaves) - 1 (root doesn't count) - 1
    TreeNode *nodes[5];
    int nodes_counter = 0;

    float max_area;
    int max_index = 0;

    while (counter < 7 && max_index != -1) {
        max_index = -1;
        max_area = -1.0;

        for (int i = 0; i < counter; i++) {
            if (!(leaves[i]->leaf)) {
                float area = leaves[i]->area;
                if (area > max_area) {
                    max_area = area;
                    max_index = i;
                }
            }
        }

        if (max_index != -1) {

            TreeNode *tmp = leaves[max_index];

            // Put this node in nodes array
            nodes[nodes_counter++] = tmp;

            // Replace the max node with its children
            leaves[max_index] = leaves[counter - 1];
            leaves[counter - 1] = tmp->left;
            leaves[counter++] = tmp->right;
        }
    }

#ifdef DEBUG_PRINT
    printf("%p counter=%d nodes_counter=%d\n", root, counter, nodes_counter);
    for (int i = 0; i < counter; i++) {
        printf("%p leaf %p\n", root, leaves[i]);
    }
    for (int i = 0; i < nodes_counter; i++) {
        printf("%p node %p\n", root, nodes[i]);
    }
#endif

    unsigned char optimal[128];

    // Call calculateOptimalCost here
    calculateOptimalTreelet(counter, leaves, optimal);

#ifdef DEBUG_PRINT
    printPartition(root, optimal, optimal[(1 << counter) - 1], (1 << counter) - 1);
#endif

    // Use complement on right tree, and use original on left tree
    unsigned char mask = (1 << counter) - 1;    // mask = max index
    int index = 0;                              // index for free nodes
    unsigned char leftIndex = mask;
    unsigned char left = optimal[leftIndex];
    restructTree(root, leaves, nodes, left, optimal, index, true, counter);

    unsigned char right = (~left) & mask;
    restructTree(root, leaves, nodes, right, optimal, index, false, counter);

    // Calculate current node's area & cost
    Bound *bound = &root->bound;
    merge_bounds(root->left->bound, root->right->bound, bound);
    root->area = getArea(bound->min_x, bound->max_x, bound->min_y,
        bound->max_y, bound->min_z, bound->max_z);
    root->cost = Ci * root->area + root->left->cost + root->right->cost;
}

/**
 * BVH Optimization kernel
 */
__global__ void kernelOptimize(int num_leaves, int *nodeCounter,
    TreeNode *treeNodes, TreeNode *treeLeaves) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= num_leaves) return;

    TreeNode *leaf = treeLeaves + index;
    
    // Handle leaf first
    // Leaf's cost is just its bounding volumn's cost
    Bound *bound = &leaf->bound;
    leaf->area = getArea(bound->min_x, bound->max_x, bound->min_y,
        bound->max_y, bound->min_z, bound->max_z);
    leaf->cost = Ct * leaf->area;

#ifdef DEBUG_PRINT
    printf("%d handled leaf\n", index);
#endif

#ifdef DEBUG_PRINT
    __syncthreads();
    if (index == 0) {
        printf("Launching Print BVH GPU... (before Optimization)\n");
        printBVH(treeNodes);
        printf("Launched Print BVH GPU... (before Optimization)\n");
    }
    __syncthreads();
#endif

    TreeNode *current = leaf->parent;
    int currentIndex = current - treeNodes;
    int res = atomicAdd(nodeCounter + currentIndex, 1);

    // Go up and handle internal nodes
    while (1) {
        if (res == 0) {
            return;
        }

#ifdef DEBUG_PRINT
        printf("%d Going to optimize %p\n", index, current);
#endif

        treeletOptimize(current);

#ifdef DEBUG_PRINT
        printf("%d Optimized %p\n", index, current);
#endif

        // If current is root, return
        if (current == treeNodes) {
            return;
        }
        current = current->parent;
        currentIndex = current - treeNodes;
        res = atomicAdd(nodeCounter + currentIndex, 1);
    }

}

void
ParallelBVH::loadScene() {
    std::cin >> this->num_geometries;
    this->spheres = (Sphere *)malloc(this->num_geometries * sizeof(Sphere));
    for (int i = 0; i < this->num_geometries; i++) {
        double rad, posx, posy, posz, emix, emiy, emiz, colx, coly, colz;
        int mat; // 1: DIFF, 2: SPEC, 3: REFR
        std::cin >> rad >> posx >> posy >> posz >> emix >> emiy >> emiz >> colx 
            >> coly >> colz >> mat;
        this->spheres[i] = Sphere(rad, Vec(posx, posy, posz), 
            Vec(emix, emiy, emiz), Vec(colx, coly, colz), 
            (mat == 1 ? DIFF : (mat == 2 ? SPEC : REFR)));
    }
}

/**
 * ParallelBVH constructor
 */
ParallelBVH::ParallelBVH() {

    // Load scene
    loadScene();

    // For internal nodes, leaf = false
    gpuErrchk( cudaMalloc(&(this->cudaDeviceTreeNodes), 
        sizeof(TreeNode) * (this->num_geometries - 1)) );
    gpuErrchk( cudaMemset(this->cudaDeviceTreeNodes, 0, 
        sizeof(TreeNode) * (this->num_geometries - 1)) );

    // For leaves, leaf = true
    gpuErrchk( cudaMalloc(&(this->cudaDeviceTreeLeaves), sizeof(TreeNode) * 
        this->num_geometries) );
    gpuErrchk( cudaMemset(this->cudaDeviceTreeLeaves, ~0, 
        sizeof(TreeNode) * this->num_geometries) );

    // Initialize morton codes
    float min_x = pos_infinity;
    float max_x = neg_infinity;
    float min_y = pos_infinity;
    float max_y = neg_infinity;
    float min_z = pos_infinity;
    float max_z = neg_infinity; 
    for (unsigned int i = 0; i < this->num_geometries; i++) {
        this->spheres[i].index = i;
        this->spheres[i].bound = Bound();
        this->spheres[i].bound.min_x = this->spheres[i].p.x - this->spheres[i].rad;
        this->spheres[i].bound.min_y = this->spheres[i].p.y - this->spheres[i].rad;
        this->spheres[i].bound.min_z = this->spheres[i].p.z - this->spheres[i].rad;
        this->spheres[i].bound.max_x = this->spheres[i].p.x + this->spheres[i].rad;
        this->spheres[i].bound.max_y = this->spheres[i].p.y + this->spheres[i].rad;
        this->spheres[i].bound.max_z = this->spheres[i].p.z + this->spheres[i].rad;
        // find min and max coordinates
        Vec p = this->spheres[i].p;
        if (p.x < min_x) min_x = p.x;
        if (p.x > max_x) max_x = p.x;
        if (p.y < min_y) min_y = p.y;
        if (p.y > max_y) max_y = p.y;
        if (p.z < min_z) min_z = p.z;
        if (p.z > max_z) max_z = p.z;
    }

    for (unsigned int i = 0; i < this->num_geometries; i++) {
        // calculate morton code
        this->spheres[i].index = i;
        this->spheres[i].morton_code = 0; 
        Vec p = this->spheres[i].p;
        // get the first 21 bits of each coordinate 
        long long a = (long long)(((p.x - min_x)/(max_x - min_x)) * CODE_OFFSET); 
        long long b = (long long)(((p.y - min_y)/(max_y - min_y)) * CODE_OFFSET); 
        long long c = (long long)(((p.z - min_z)/(max_z - min_z)) * CODE_OFFSET);
        // combine into 63 bits morton code
        for (unsigned int j = 0; j < CODE_LENGTH; j++) { 
          this->spheres[i].morton_code |=
          (((((a >> (CODE_LENGTH - 1 - j))) & 1) << ((CODE_LENGTH - j) * 3 - 1)) |
           ((((b >> (CODE_LENGTH - 1 - j))) & 1) << ((CODE_LENGTH - j) * 3 - 2)) |
           ((((c >> (CODE_LENGTH - 1 - j))) & 1) << ((CODE_LENGTH - j) * 3 - 3)) );
      }
    }

    // Scene geometries
    gpuErrchk( cudaMalloc(&(this->cudaDeviceGeometries), 
        sizeof(Sphere) * this->num_geometries) );
    gpuErrchk( cudaMemcpy(this->cudaDeviceGeometries, this->spheres,
        sizeof(Sphere) * this->num_geometries, cudaMemcpyHostToDevice) );

}

/**
 * ParallelBVH destructor
 */
ParallelBVH::~ParallelBVH() {
    if (cudaDeviceTreeNodes) {
        cudaFree(cudaDeviceTreeNodes);
    }
    if (cudaDeviceTreeLeaves) {
        cudaFree(cudaDeviceTreeLeaves);
    }
    if (cudaDeviceGeometries) {
        cudaFree(cudaDeviceGeometries);
    }
    if (this->spheres) {
        cudaFree(this->spheres);
    }
}

/**
 * ParallelBVH::constructBVHTree
 * Construct the BVH tree to be used for ray tracing later.
 * Required that a radix tree is constructed
 */
void
ParallelBVH::constructBVHTree() {

    // Sort geometries
    size_t i;

    // initialize host variables
    thrust::host_vector<long long> H_keys(this->num_geometries);
    thrust::host_vector<int> H_values(this->num_geometries);

    // place data into host variables
    for (i = 0; i < this->num_geometries; i++) {
        H_keys[i] = this->spheres[i].morton_code;
        H_values[i] = i;
    }

#ifdef DEBUG_PRINT
    std::cout << "Placed geometries into host variables...\n";
#endif

    // copy unsorted data from host to device
    thrust::device_vector<long long> D_keys = H_keys;
    thrust::device_vector<int> D_values = H_values;

    // sort the data
    thrust::sort_by_key(D_keys.begin(), D_keys.end(), D_values.begin());

#ifdef DEBUG_PRINT
    std::cout << "Sorted geometries...\n";
#endif

    // extract device memory pointer of sorted array
    int *sorted_geometry_indices = thrust::raw_pointer_cast(D_values.data());

    // nodeCounter makes sure that only 1 thread get to work on a node
    // in BVH construction kernel
    int *nodeCounter;
#ifdef DEBUG_CHECK
    gpuErrchk( cudaMalloc(&nodeCounter, sizeof(int) * (this->num_geometries)) );
    gpuErrchk( cudaMemset(nodeCounter, 0, sizeof(int) * (this->num_geometries)) );
#else
    cudaMalloc(&nodeCounter, sizeof(int) * (this->num_geometries));
    cudaMemset(nodeCounter, 0, sizeof(int) * (this->num_geometries));
#endif

    // Configure GPU running parameters
    dim3 blockDim(BLOCK_SIZE, 1);
    dim3 gridDim(this->num_geometries + (BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

#ifdef DEBUG_PRINT
    std::cout << "Launching BVH Construction GPU...\n";
#endif

    // Launch BVH contruction kernel
    kernelConstructBVHTree<<<gridDim, blockDim>>>(this->num_geometries, 
        cudaDeviceTreeNodes, cudaDeviceTreeLeaves, nodeCounter, 
        sorted_geometry_indices, cudaDeviceGeometries);

#ifdef DEBUG_PRINT
    std::cout << "Launched BVH Construction GPU...\n";
#endif

#ifdef DEBUG_CHECK
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaFree(nodeCounter) );
#else
    cudaDeviceSynchronize();
    cudaFree(nodeCounter);
#endif
}

/**
 * ParallelBVH::constructRadixTree
 * Construct a radix tree. Required for the construction of BVH tree.
 * Described in karras2012 paper.
 */
void
ParallelBVH::constructRadixTree() {

    // Setup number of iterations for each thread
    int internalNodes = this->num_geometries - 1;

    // Configure GPU running parameters
    dim3 blockDim(BLOCK_SIZE, 1);
    dim3 gridDim((internalNodes + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    
#ifdef DEBUG_PRINT
    std::cout << "Launching Radix Tree Construction GPU...\n";
#endif

    // Launch the radix tree construction kernel
    kernelConstructRadixTree<<<gridDim, blockDim>>>(internalNodes, 
        cudaDeviceTreeNodes, cudaDeviceTreeLeaves);

#ifdef DEBUG_CHECK
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
#else
    cudaDeviceSynchronize();
#endif

#ifdef DEBUG_PRINT
    std::cout << "Launched Radix Tree Construction GPU...\n";
#endif
}

/**
 * ParallelBVH::optimize
 * Use treelet reconstruction algorithm described in karras2013hpg_paper.pdf
 * to optimize BVH tree
 */
void
ParallelBVH::optimize() {

    // nodeCounter makes sure that only 1 thread get to work on a node
    // in BVH construction kernel
    int *nodeCounter;
#ifdef DEBUG_CHECK
    gpuErrchk( cudaMalloc(&nodeCounter, sizeof(int) * (this->num_geometries)) );
    gpuErrchk( cudaMemset(nodeCounter, 0, sizeof(int) * (this->num_geometries)) );
#else
    cudaMalloc(&nodeCounter, sizeof(int) * (this->num_geometries));
    cudaMemset(nodeCounter, 0, sizeof(int) * (this->num_geometries));
#endif

    // Configure GPU running parameters
    dim3 blockDim(BLOCK_SIZE, 1);
    dim3 gridDim((this->num_geometries + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

#ifdef DEBUG_PRINT
    std::cout << "Launching BVH Optimization GPU...\n";
#endif

    // Launch the optimize kernel
    kernelOptimize<<<gridDim, blockDim>>>(this->num_geometries, nodeCounter,
        cudaDeviceTreeNodes, cudaDeviceTreeLeaves);

#ifdef DEBUG_CHECK
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
#else
    cudaDeviceSynchronize();
#endif

#ifdef DEBUG_PRINT
    std::cout << "Launched BVH Optimization GPU...\n";
#endif

#ifdef DEBUG_CHECK
    gpuErrchk( cudaFree(nodeCounter) );
#else
    cudaDeviceSynchronize();
    cudaFree(nodeCounter);
#endif

#ifdef DEBUG_PRINT
    std::cout << "Launching Print BVH GPU... (after Optimization)\n";

    kernelPrintBVH<<<1, 1>>>(cudaDeviceTreeNodes);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    std::cout << "Launched Print BVH GPU... (after Optimization)\n";
#endif

}

/**
 * ParallelBVH::raytrace
 * Run GPU ray tracing using the BVH created later.
 * Required that constructRadixTree and constructBVHTree were called earlier
 * to work
 */
void
ParallelBVH::raytrace(Vec *buffer, int width, int height,
    unsigned int samps, const Ray &cam, Vec &cx, Vec &cy) {

    int total_pixels = width * height;

    curandState *states;

#ifdef DEBUG_CHECK
    gpuErrchk( cudaMalloc(&states, 4 * total_pixels * sizeof(curandState)) );
#else
    cudaMalloc(&states, 4 * total_pixels * sizeof(curandState));
#endif

    // Allocate pixel array on GPU
    Vec *deviceSubpixelBuffer;
#ifdef DEBUG_CHECK
    gpuErrchk( cudaMalloc(&deviceSubpixelBuffer, 4 * total_pixels * sizeof(Vec)) );
#else
    cudaMalloc(&deviceSubpixelBuffer, 4 * total_pixels * sizeof(Vec));
#endif

    // Configure GPU running parameters
    int total_threads = total_pixels * 4;
    dim3 blockDim(BLOCK_SIZE, 1);
    dim3 gridDim((total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

#ifdef DEBUG_PRINT
    std::cout << "Launching Ray Tracing GPU...\n";
#endif

    clock_t t1, t2;
    float mseconds;

    t1 = clock();
    // Launch the ray trace kernel
    kernelRayTrace<<<gridDim, blockDim>>>(states, deviceSubpixelBuffer, width, 
        height, samps, cudaDeviceGeometries, cudaDeviceTreeNodes, cam, cx, cy);

#ifdef DEBUG_CHECK
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
#else
    cudaDeviceSynchronize();
#endif
    
    t2 = clock();
    mseconds = ((float)(t2 - t1))/ CLOCKS_PER_SEC;
    std::cout << "Ray Tracing took " << mseconds << "\n";

#ifdef DEBUG_PRINT
    std::cout << "Launched Ray Tracing GPU...\n";
#endif

    // Get the result
    Vec *devicePixelBuffer;
#ifdef DEBUG_CHECK
    gpuErrchk( cudaMalloc(&devicePixelBuffer, total_pixels * sizeof(Vec)) );
#else
    cudaMalloc(&devicePixelBuffer, total_pixels * sizeof(Vec));
#endif

#ifdef DEBUG_PRINT
    std::cout << "Launching Get Result GPU...\n";
#endif

    // Configure GPU running parameters
    dim3 gridDim2((total_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    kernelGetResult<<<gridDim2, blockDim>>>(deviceSubpixelBuffer,
        devicePixelBuffer, width, height);

#ifdef DEBUG_CHECK
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
#else
    cudaDeviceSynchronize();
#endif

#ifdef DEBUG_PRINT
    std::cout << "Launching Get Result GPU...\n";
#endif


#ifdef DEBUG_CHECK
    // Copy back the pixel results to host
    gpuErrchk( cudaMemcpy(buffer, devicePixelBuffer, total_pixels * sizeof(Vec), 
        cudaMemcpyDeviceToHost) );

    // Free resources
    gpuErrchk( cudaFree(states) );
    gpuErrchk( cudaFree(devicePixelBuffer) );
    gpuErrchk( cudaFree(deviceSubpixelBuffer) );
#else
    // Copy back the pixel results to host
    cudaMemcpy(buffer, devicePixelBuffer, total_pixels * sizeof(Vec), 
        cudaMemcpyDeviceToHost);

    // Free resources
    cudaFree(states);
    cudaFree(devicePixelBuffer);
    cudaFree(deviceSubpixelBuffer);
#endif
}
