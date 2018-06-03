// -*- C++ -*-
#ifndef MAIN1_CUDA_CUH
#define MAIN1_CUDA_CUH

#include <cuda_runtime.h>
#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#endif
#include <cuda_gl_interop.h>

// Create/delete data in VBOs
void createVBOs(GLuint* vbo);
void deleteVBOs(GLuint* vbo);

// Handles global variables
int getNumPoints();
void setNumPoints(int n);
int getDimension();
void setDimension(int n);
void setCUDA();
void changeFunction();

#endif // MAIN1_CUDA_CUH
