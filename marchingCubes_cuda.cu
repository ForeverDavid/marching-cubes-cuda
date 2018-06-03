///////////////////////////////////////////////////////////////////////////////
// MARCHING CUBES															 //
///////////////////////////////////////////////////////////////////////////////
// CS179 - SPRING 2014
// Final project
// Victor Ceballos Inza

// This file contains the CUDA kernels used in the algorithm, as well as the
// functions that initialize and clean up the VBOs.

///////////////////////////////////////////////////////////////////////////////
// Includes																	 //
///////////////////////////////////////////////////////////////////////////////
#include "marchingCubes_cuda.cuh"
#include <helper_math.h>
#include <stdio.h>

///////////////////////////////////////////////////////////////////////////////
// Declarations																 //
///////////////////////////////////////////////////////////////////////////////
#define BLOCK_SIZE 512
#define PI 3.141592654f

#define gpuErrchk(ans) { gpuAssert((ans), (char*)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char* file, int line,
		bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
				line);
		if (abort)
			exit(code);
	}
}

// Rendering variables
float xmax = 10.0f;
float xmin = -10.0f;
int numPoints = 3;
int dim = 2;
int func = 0;

// Flag to toggle CUDA usage
int cuda = 1;


///////////////////////////////////////////////////////////////////////////////
// Marching cubes table data												 //
///////////////////////////////////////////////////////////////////////////////

__device__
int pointTable[4][1] = {
		{-1}, {0}, {0}, {-1},
};

__device__
int lineTable[16][4] = {
		{-1, -1, -1, -1},
		{ 0,  3, -1, -1},
		{ 0,  1, -1, -1},
		{ 3,  1, -1, -1},
		{ 1,  2, -1, -1},
		{ 0,  1,  3,  2},
		{ 0,  2, -1, -1},
		{ 3,  2, -1, -1},
		{ 3,  2, -1, -1},
		{ 0,  2, -1, -1},
		{ 0,  3,  1,  2},
		{ 1,  2, -1, -1},
		{ 3,  1, -1, -1},
		{ 0,  1, -1, -1},
		{ 0,  3, -1, -1},
		{-1, -1, -1, -1},
};

__device__
int triangleTable[256][15] = {
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1},
	{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1},
	{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1},
	{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1},
	{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1},
	{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1},
	{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1},
	{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1},
	{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1},
	{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1},
	{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1},
	{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1},
	{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1},
	{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1},
	{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1},
	{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1},
	{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1},
	{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1},
	{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1},
	{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1},
	{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1},
	{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1},
	{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1},
	{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1},
	{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1},
	{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1},
	{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1},
	{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1},
	{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0},
	{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0},
	{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1},
	{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1},
	{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1},
	{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1},
	{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1},
	{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1},
	{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1},
	{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1},
	{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1},
	{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1},
	{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1},
	{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9},
	{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1},
	{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1},
	{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6},
	{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1},
	{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11},
	{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7},
	{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1},
	{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1},
	{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1},
	{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1},
	{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1},
	{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1},
	{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1},
	{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1},
	{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1},
	{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1},
	{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1},
	{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1},
	{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1},
	{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1},
	{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1},
	{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1},
	{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1},
	{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1},
	{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9},
	{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1},
	{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1},
	{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7},
	{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11},
	{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1},
	{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6},
	{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1},
	{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1},
	{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1},
	{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1},
	{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1},
	{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1},
	{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1},
	{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1},
	{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1},
	{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1},
	{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1},
	{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1},
	{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1},
	{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1},
	{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1},
	{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3},
	{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1},
	{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1},
	{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1},
	{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1},
	{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1},
	{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3},
	{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1},
	{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1},
	{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1},
	{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1},
	{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1},
	{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6},
	{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1},
	{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1},
	{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8},
	{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1},
	{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4},
	{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10},
	{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1},
	{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1},
	{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1},
	{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1},
	{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10},
	{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5},
	{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1},
	{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1},
	{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1},
	{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8},
	{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6},
	{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1},
	{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1},
	{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1},
	{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1},
	{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1},
	{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1},
	{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2},
	{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1},
	{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1},
	{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1},
	{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2},
	{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1},
	{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1},
	{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1},
	{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1},
	{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1},
	{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4},
	{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1},
	{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11},
	{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5},
	{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1},
	{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1},
	{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9},
	{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1},
	{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1},
	{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1},
	{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1},
	{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4},
	{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1},
	{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3},
	{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1},
	{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1},
	{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1},
	{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7},
	{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10},
	{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1},
	{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1},
	{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1},
	{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1},
	{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1},
	{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1},
	{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
};

__device__
int cube_edgeToVerts[12][2] = {
	{0,1}, {1,2}, {2,3}, {3,0},
	{4,5}, {5,6}, {6,7}, {7,4},
	{0,4}, {1,5}, {2,6}, {3,7},
};

///////////////////////////////////////////////////////////////////////////////
// Surface functions														 //
///////////////////////////////////////////////////////////////////////////////

// Surface to be rendered in 1D
__device__ __host__
int function1D(float4& point, int func)
{
	float fun; int flag;
	switch (func) {

	case 0:
		fun = point.x * point.x;
		flag = (fun < 9);
		break;

	case 1:
		fun = point.x;
		flag = (fun < 0) and (fun > -10);
		break;

	case 2:
		fun = point.x;
		flag = ( (fun > -9) and (fun < -8) ) or ( (fun > 3) and (fun < 7) );
		break;
	}

	return flag;
}

// Surface to be rendered in 2D
__device__ __host__
int function2D(float4& point, int func)
{
	float fun; int flag;
	float c1,c2,c3,c4,c5;
	switch (func) {

	case 0:
		fun = point.x * point.x + point.y * point.y;
		flag = (fun < 9);
		break;

	case 1:
		c1 = (point.x+9)*(point.x+9) + (point.y-9)*(point.y-9);
		c2 = (point.x-0)*(point.x-0) + (point.y-0)*(point.y-0);
		c3 = (point.x+4)*(point.x+4) + (point.y-6)*(point.y-6);
		c4 = (point.x-5)*(point.x-5) + (point.y-5)*(point.y-5);
		c5 = (point.x+7)*(point.x+7) + (point.y+13)*(point.y+13);
		flag = (c1 < 1) or (c2 < 2.25) or (c3 < 4) or (c4 < 25)
			or (c5 < 64 and -10<point.x and -10<point.y)
			or (2<point.x and point.x<4 and -5<point.y and point.y<-3)
			or (point.y < point.x - 15.0 and point.x<10 and -10<point.y);
		break;

	case 2:
		fun = log((double)point.x);
		flag = abs(point.y) < abs(fun) and point.x > 0 and point.x < 10;
		break;
	}

	return flag;
}

// Surface to be rendered in 3D
__device__ __host__
int function3D(float4& point, int func)
{
	float fun; int flag;
	switch (func) {

	case 0:
		fun = point.x * point.x + point.y * point.y + point.z * point.z;
		flag = (fun < 9);
		break;

	case 1:
		fun = point.x * point.x / 5.0 + point.y * point.y / 3.0
			- point.z * point.z / 7.0;
		flag = (fun < 5);
		break;

	case 2:
		fun = point.x * point.x / 10.0 - point.y * point.y / 3.0
			- point.z / 2.0;
		flag = (fun < 0);
		break;
	}

	return flag;
}


///////////////////////////////////////////////////////////////////////////////
// CUDA kernels																 //
///////////////////////////////////////////////////////////////////////////////

// This kernel checks whether each point lies within the desired surface.
__global__
void points_kernel(float4* points, int size, int dim, int func)
{
	// Get unique thread id
	unsigned int globalID = blockIdx.x * blockDim.x + threadIdx.x;

	// Check whether the point lies within.
	// Fourth coordinate represent containment.
	switch (dim) {

	case 1:

		for (int k = globalID; k < size; k += gridDim.x * blockDim.x) {
			float4 pt = points[k];
			points[k].w = function1D(pt, func);
		}
		break;

	case 2:

		for (int k = globalID; k < size * size; k += gridDim.x * blockDim.x) {
			float4 pt = points[k];
			points[k].w = function2D(pt, func);
		}
		break;

	case 3:

		for (int k = globalID; k < size * size * size;
				k += gridDim.x * blockDim.x) {
			float4 pt = points[k];
			points[k].w = function3D(pt, func);
		}
		break;
	}
}

// This kernel classifies each interval in the grid.
__global__
void kernel1D(float4* points, float4* geom, int size)
{
	// Get unique thread ID, this is the point ID
	unsigned int globalID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int id = globalID; id < size; id += gridDim.x * blockDim.x) {

		// Point ID equals interval ID
		if (id < (size - 1) ) {

			// Get the vertices of this interval
			float4 verts[2];
			verts[0] = points[id];
			verts[1] = points[id + 1];

			// Obtain the type of this interval
			int type = 0;
			for (int l = 0; l < 2; l++) {
				type += verts[l].w * pow((double)2,(double)l);
			}

			// Get the configuration for this type of interval from the table
			// and generate the points accordingly
			int* config = pointTable[type];
			int e = config[0];
			if (e != -1) {
				geom[id] = ( verts[e%2] + verts[(e+1)%2] ) * (0.5f);
				geom[id].w = 1.0f;
			}

		}
	}
}

// This kernel classifies each square in the grid.
__global__
void kernel2D(float4* points, float4* geom, int size)
{
	// Get unique thread ID, this is the point ID
	unsigned int globalID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int id = globalID; id < size * size; id += gridDim.x * blockDim.x) {

		// Transform point ID to square ID
		int j = (int) floor((double) (id / size));
		int idx = id - j;

		if (idx < (size - 1) * (size - 1)) {

			// Get the vertices of this square
			float4 verts[4];
			verts[0] = points[id];
			verts[1] = points[id + 1];
			verts[2] = points[id + size + 1];
			verts[3] = points[id + size];

			// Obtain the type of this square
			int type = 0;
			for (int l = 0; l < 4; l++) {
				type += verts[l].w * pow((double)2,(double)l);
			}

			// Get the configuration for this type of square from the table
			// and generate the lines accordingly
			int* config = lineTable[type];
			int e;
			for (int l = 0; l < 4; l++) {
				e = config[l];
				if (e != -1) {
					geom[4*idx + l] = ( verts[e%4] + verts[(e+1)%4] ) * (0.5f);
					geom[4*idx + l].w = 1.0f;
				} else { break; }
			}

		}
	}
}

// This kernel classifies each cube in the grid.
__global__
void kernel3D(float4* points, float4* geom, int size)
{
	// Get unique thread ID, this is the point ID
	unsigned int globalID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int id = globalID; id < size * size * size; id += gridDim.x * blockDim.x) {

		// Transform point ID to cube ID
		int j = (int) ( (int) floor((double) (id / size)) % size );
		int k = (int) floor((double) (id / (size*size)));
		int idx = id - j + k - 2 * k * size;

		if (idx < (size - 1) * (size - 1) * (size - 1)) {

			// Get the vertices of this cube
			float4 verts[8];
			verts[0] = points[id];
			verts[1] = points[id + 1];
			verts[2] = points[id + size + 1];
			verts[3] = points[id + size];

			verts[4] = points[id + size*size];
			verts[5] = points[id + size*size + 1];
			verts[6] = points[id + size*size + size + 1];
			verts[7] = points[id + size*size + size];

			// Obtain the type of this cube
			int type = 0;
			for (int l = 0; l < 8; l++) {
				type += verts[l].w * pow((double)2,(double)l);
			}

			// Get the configuration for this type of cube from the table
			// and generate the triangles accordingly
			int* config = triangleTable[type];
			int e, e0, e1;
			for (int l = 0; l < 15; l++) {
				e = config[l];
				e0 = cube_edgeToVerts[e][0]; e1 = cube_edgeToVerts[e][1];
				if (e != -1) {
					geom[15*idx + l] = ( verts[e0] + verts[e1] ) * (0.5f);
					geom[15*idx + l].w = 1.0f;
				} else { break; }
			}

		}
	}
}


///////////////////////////////////////////////////////////////////////////////
// Run the CUDA part of the computation										 //
///////////////////////////////////////////////////////////////////////////////
void runCuda(GLuint *vbo)
{
	// Map OpenGL buffer object for writing from CUDA
	float4* dev_points;
	float4* dev_geometry;

	// Map OpenGL buffers to CUDA
	cudaGLMapBufferObject((void**) &dev_points, vbo[1]);
	cudaGLMapBufferObject((void**) &dev_geometry, vbo[2]);

	// Choose a block size and a grid size
	const unsigned int threadsPerBlock = BLOCK_SIZE;
	const unsigned int maxBlocks = 50;
	unsigned int blocks;

	// Execute CUDA kernels
	switch (dim) {

	case 1:

		blocks = min(maxBlocks,
				(int) ceil(numPoints / (float) threadsPerBlock));

		// Check for containment of vertices
		points_kernel<<<blocks, threadsPerBlock>>>
				(dev_points, numPoints, dim, func);

		// Obtain the edges from the data table
		kernel1D<<<blocks, threadsPerBlock>>>
				(dev_points, dev_geometry, numPoints);

		break;

	case 2:

		blocks = min(maxBlocks,
				(int) ceil(numPoints * numPoints / (float) threadsPerBlock));

		// Check for containment of vertices
		points_kernel<<<blocks, threadsPerBlock>>>
				(dev_points, numPoints, dim, func);

		// Obtain the edges from the data table
		kernel2D<<<blocks, threadsPerBlock>>>
				(dev_points, dev_geometry, numPoints);

		break;

	case 3:

		blocks = min(maxBlocks,
				(int) ceil(
						numPoints * numPoints * numPoints
								/ (float) threadsPerBlock));

		// Check for containment of vertices
		points_kernel<<<blocks, threadsPerBlock>>>
				(dev_points, numPoints, dim, func);

		// Obtain the triangles from the data table
		kernel3D<<<blocks, threadsPerBlock>>>
				(dev_points, dev_geometry, numPoints);

		break;
	}

	// Unmap buffer objects from CUDA
	cudaGLUnmapBufferObject(vbo[1]);
	cudaGLUnmapBufferObject(vbo[2]);
}


///////////////////////////////////////////////////////////////////////////////
// Vertex Buffer Objects													 //
///////////////////////////////////////////////////////////////////////////////

// Initialize 1D data
void createData1D(float4* points, float4* grid, float4* geom)
{
	// Initialize points data.
	float delta = (xmax - xmin) / (numPoints - 1);
	for (int i = 0; i < numPoints; i++) {

		// Set initial position data
		points[i].x = xmin + delta * i;
		points[i].y = 0.0f;
		points[i].z = 0.0f;
		points[i].w = 1.0f;
	}

	// Initialize grid data.
	for (int i = 0; i < (numPoints - 1); i++) {

		// Set initial position data
		grid[2*i+0] = points[i];
		grid[2*i+1] = points[i+1];
	}

	// Initialize geometry data.
	float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	for (int k = 0; k < (numPoints - 1); k++) {
		geom[k] = zero;
	}
}

// Initialize 2D data
void createData2D(float4* points, float4* grid, float4* geom)
{
	// Initialize points data.
	float delta = (xmax - xmin) / (numPoints - 1);
	for (int i = 0; i < numPoints; i++) {
		for (int j = 0; j < numPoints; j++) {

			int idx = i + j * numPoints;

			// Set initial position data
			points[idx].x = xmin + delta * i;
			points[idx].y = xmax - delta * j;
			points[idx].z = 0.0f;
			points[idx].w = 1.0f;
		}
	}

	// Initialize grid data.
	for (int i = 0; i < (numPoints - 1); i++) {
		for (int j = 0; j < (numPoints - 1); j++) {

			int idx_pt = i + j * numPoints;
			int idx_sq = idx_pt - j;

			// Set initial position data
			grid[4 * idx_sq + 0] = points[idx_pt];
			grid[4 * idx_sq + 1] = points[idx_pt + 1];
			grid[4 * idx_sq + 2] = points[idx_pt + numPoints + 1];
			grid[4 * idx_sq + 3] = points[idx_pt + numPoints];
		}
	}

	// Initialize geometry data.
	float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	for (int k = 0; k < (numPoints - 1) * (numPoints - 1) * 4; k++) {
		geom[k] = zero;
	}
}

// Initialize 3D data
void createData3D(float4* points, float4* grid, float4* geom)
{
	// Initialize points data.
	float delta = (xmax - xmin) / (numPoints - 1);
	for (int i = 0; i < numPoints; i++) {
		for (int j = 0; j < numPoints; j++) {
			for (int k = 0; k < numPoints; k++) {

				int idx = i + j * numPoints + k * numPoints * numPoints;

				// Set initial position data
				points[idx].x = xmin + delta * i;
				points[idx].y = xmax - delta * j;
				points[idx].z = xmin + delta * k;
				points[idx].w = 1.0f;
			}
		}
	}

	// Initialize grid data.
	for (int i = 0; i < (numPoints - 1); i++) {
		for (int j = 0; j < (numPoints - 1); j++) {
			for (int k = 0; k < (numPoints - 1); k++) {

				int idx_pt = i + j * numPoints + k * numPoints * numPoints;
				int idx_sq = idx_pt - j + k - 2 * k * numPoints;

				// Set initial position data
				grid[16 * idx_sq + 0] = points[idx_pt];
				grid[16 * idx_sq + 1] = points[idx_pt+1];
				grid[16 * idx_sq + 2] = points[idx_pt+numPoints+1];
				grid[16 * idx_sq + 3] = points[idx_pt+numPoints];

				grid[16 * idx_sq + 4] = points[idx_pt+numPoints*numPoints];
				grid[16 * idx_sq + 5] = points[idx_pt+numPoints*numPoints+1];
				grid[16 * idx_sq + 6] = points[idx_pt+numPoints*numPoints+numPoints+1];
				grid[16 * idx_sq + 7] = points[idx_pt+numPoints*numPoints+numPoints];

				grid[16 * idx_sq + 8] = points[idx_pt];
				grid[16 * idx_sq + 9] = points[idx_pt+1];
				grid[16 * idx_sq + 10] = points[idx_pt+numPoints*numPoints+1];
				grid[16 * idx_sq + 11] = points[idx_pt+numPoints*numPoints];

				grid[16 * idx_sq + 12] = points[idx_pt+numPoints];
				grid[16 * idx_sq + 13] = points[idx_pt+numPoints+1];
				grid[16 * idx_sq + 14] = points[idx_pt+numPoints*numPoints+numPoints+1];
				grid[16 * idx_sq + 15] = points[idx_pt+numPoints*numPoints+numPoints];

			}
		}
	}

	// Initialize geometry data.
	float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	for (int k = 0; k < (numPoints - 1) * (numPoints - 1) * (numPoints - 1) * 15; k++) {
		geom[k] = zero;
	}
}

// Create VBOs
void createVBOs(GLuint* vbo)
{
	// Create VBOs.
	glGenBuffers(3, vbo);

	// Initialize points and grid
	unsigned int points_size;
	float4* points;
	unsigned int grid_size;
	float4* grid;
	unsigned int geom_size;
	float4* geom;

	switch (dim) {

	case 1:

		// Allocate memory
		points_size = numPoints * sizeof(float4);
		points = (float4*) malloc(points_size);
		grid_size = (numPoints - 1) * 2 * sizeof(float4);
		grid = (float4*) malloc(grid_size);
		geom_size = (numPoints - 1) * sizeof(float4);
		geom = (float4*) malloc(geom_size);
		// Initialize data
		createData1D(points, grid, geom);
		break;

	case 2:

		// Allocate memory
		points_size = numPoints * numPoints * sizeof(float4);
		points = (float4*) malloc(points_size);
		grid_size = (numPoints - 1) * (numPoints - 1) * 4 * sizeof(float4);
		grid = (float4*) malloc(grid_size);
		geom_size = (numPoints - 1) * (numPoints - 1) * 4 * sizeof(float4);
		geom = (float4*) malloc(geom_size);
		// Initialize data
		createData2D(points, grid, geom);
		break;

	case 3:

		// Allocate memory
		points_size = numPoints * numPoints * numPoints * sizeof(float4);
		points = (float4*) malloc(points_size);
		grid_size = (numPoints - 1) * (numPoints - 1) * (numPoints - 1) * 16
				* sizeof(float4);
		grid = (float4*) malloc(grid_size);
		geom_size = (numPoints - 1) * (numPoints - 1) * (numPoints - 1) * 15
				* sizeof(float4);
		geom = (float4*) malloc(geom_size);
		// Initialize data
		createData3D(points, grid, geom);
		break;
	}

	// Activate VBO id to use.
	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

	// Upload data to video card.
	glBufferData(GL_ARRAY_BUFFER, grid_size, grid, GL_DYNAMIC_DRAW);

	// Activate VBO id to use.
	glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);

	// Upload data to video card.
	glBufferData(GL_ARRAY_BUFFER, points_size, points, GL_DYNAMIC_DRAW);

	// Register buffer objects with CUDA
	gpuErrchk(cudaGLRegisterBufferObject(vbo[1]));

	// Activate VBO id to use.
	glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);

	// Upload data to video card.
	glBufferData(GL_ARRAY_BUFFER, geom_size, geom, GL_DYNAMIC_DRAW);

	// Register buffer objects with CUDA
	gpuErrchk(cudaGLRegisterBufferObject(vbo[2]));

	// Free temporary data
	free(points); free(grid); free(geom);

	// Release VBOs with ID 0 after use.
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Execute the algorithm, if asked
	if (cuda) { runCuda(vbo); }

}

// Delete VBOs
void deleteVBOs(GLuint* vbo)
{
	// Delete VBOs
	glBindBuffer(1, vbo[0]);
	glDeleteBuffers(1, &vbo[0]);
	glBindBuffer(1, vbo[1]);
	glDeleteBuffers(1, &vbo[1]);
	glBindBuffer(1, vbo[2]);
	glDeleteBuffers(1, &vbo[2]);

	// Unregister buffer objects with CUDA
	gpuErrchk(cudaGLUnregisterBufferObject(vbo[1]));
	gpuErrchk(cudaGLUnregisterBufferObject(vbo[2]));

	// Free VBOs
	*vbo = 0;
}


///////////////////////////////////////////////////////////////////////////////
// Gets/sets the number of vertices											 //
///////////////////////////////////////////////////////////////////////////////
int getNumPoints()
{
	return numPoints;
}
void setNumPoints(int n)
{
	numPoints = n;
}

///////////////////////////////////////////////////////////////////////////////
// Gets/sets the dimension													 //
///////////////////////////////////////////////////////////////////////////////
int getDimension()
{
	return dim;
}
void setDimension(int n)
{
	dim = n;
}

///////////////////////////////////////////////////////////////////////////////
// Sets the GPU usage														 //
///////////////////////////////////////////////////////////////////////////////
void setCUDA()
{
	cuda = 1 - cuda;
}

///////////////////////////////////////////////////////////////////////////////
// Changes the function of the surface to render							 //
///////////////////////////////////////////////////////////////////////////////
void changeFunction()
{
	func = (func+1)%3;
}


