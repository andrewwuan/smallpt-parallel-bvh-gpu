// Parallel BVH Interface

#ifndef _PARALLEL_BVH_H_
#define _PARALLEL_BVH_H_

//#define DEBUG_CHECK
//#define DEBUG_PRINT

#include <float.h>

const float neg_infinity = FLT_MIN;
const float pos_infinity = FLT_MAX;

class Queue // as a doubly-linked list
{
private:
	typedef struct Node {
		struct Node *prev;
		struct Node *next;
		void *v;
	} Node;

	Node *head;
	Node *tail;
public:

	__device__ __host__ Queue() {
		head = NULL;
		tail = NULL;
	}

	__device__ __host__ void push(void *v) {
		Node *n = new Node();
		n->v = v;
		if (head == NULL) {
			n->prev = NULL;
			n->next = NULL;
			head = n;
			tail = n;
		} else {
			n->prev = NULL;
			n->next = head;
			if (head != NULL) {
				head->prev = n;
			}
			head = n;
		}
	}

	__device__ __host__ void pop() {
		if (head != NULL) {
			if (head == tail) {
				delete head;
				head = NULL;
				tail = NULL;
			} else {
				Node *subtail = tail->prev;
				delete tail;
				subtail->next = NULL;
				tail = subtail;
			}
		}
	}

	__device__ __host__ void *last() {
		if (tail != NULL) {
			return tail->v;
		} else {
			return NULL;
		}
	}

	__device__ __host__ bool empty() {
		return head == NULL;
	}
};

class Bound
{
public:
	__device__ __host__ Bound() {
		min_x = pos_infinity;
		max_x = neg_infinity;
		min_y = pos_infinity;
		max_y = neg_infinity;
		min_z = pos_infinity;
		max_z = neg_infinity;
	};

	__device__ __host__ Bound(float x1, float x2, float y1, float y2, float z1, float z2) {
		min_x = x1;
		max_x = x2;
		min_y = y1;
		max_y = y2;
		min_z = z1;
		max_z = z2;
	};

	// the minimum and maximum of the coordinate for the geometry 
	// so that the box completely contains the geometry
	float min_x, max_x, min_y, max_y, min_z, max_z;

	__device__ __host__ void merge_bounds(const Bound& b1, const Bound& b2, Bound *b3);
};

struct Vec {        // Usage: time ./smallpt 5000 && xv image.ppm
  double x, y, z;                  // position, also color (r,g,b)
  __device__ __host__ Vec(double x_=0, double y_=0, double z_=0){ x=x_; y=y_; z=z_; }
  __device__ __host__ Vec operator+(const Vec &b) const { return Vec(x+b.x,y+b.y,z+b.z); }
  __device__ __host__ Vec operator-(const Vec &b) const { return Vec(x-b.x,y-b.y,z-b.z); }
  __device__ __host__ Vec operator*(double b) const { return Vec(x*b,y*b,z*b); }
  __device__ __host__ Vec mult(const Vec &b) const { return Vec(x*b.x,y*b.y,z*b.z); }
  __device__ __host__ Vec& norm(){ return *this = *this * (1/sqrt(x*x+y*y+z*z)); }
  __device__ __host__ double dot(const Vec &b) const { return x*b.x+y*b.y+z*b.z; } // cross:
  __device__ __host__ Vec operator%(Vec&b){return Vec(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);}
};

struct Ray { Vec o, d; __device__ __host__ Ray(Vec o_, Vec d_) : o(o_), d(d_) {} };

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

struct Sphere {
  double rad;       // radius
  Vec p, e, c;      // position, emission, color
  Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
  Bound bound;
  int index;
  long long morton_code;
  __device__ __host__ Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_):
	rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
  __device__ __host__ double intersect(const Ray &r) const { // returns distance, 0 if nohit
	Vec op = p-r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
	double t, eps=1e-4, b=op.dot(r.d), det=b*b-op.dot(op)+rad*rad;
	if (det<0) return 0; else det=sqrt(det);
	return (t=b-det)>eps ? t : ((t=b+det)>eps ? t : 0);
  }
};

typedef struct TreeNode {

	int min;
	int max;

	Sphere *sphere;
	bool leaf;

	TreeNode *left;
	TreeNode *right;
	TreeNode *parent;

	Bound bound;
	float cost;
	float area;

} TreeNode;


class ParallelBVH {

private:
	// Host side
	Sphere *spheres;

	int num_geometries;

	// Device side
	TreeNode *cudaDeviceTreeNodes;
	TreeNode *cudaDeviceTreeLeaves;
	Sphere *cudaDeviceGeometries;
	
public:

	// n: number of geometry centroids
	ParallelBVH();
	virtual ~ParallelBVH();

	// Load scene
	void loadScene();

	// Construct the radix tree
	void constructRadixTree();

	// Construct the BVH tree
	void constructBVHTree();

	// Use treelet reconstruction
	void optimize();

	void raytrace(Vec *buffer, int width, int height, 
	unsigned int samps, const Ray &cam, Vec &cx, Vec &cy);
};

#endif /* _PARALLEL_BVH_H_ */
