///////////////////////////////////////////////////////////////////////////////
// MARCHING CUBES															 //
///////////////////////////////////////////////////////////////////////////////
// CS179 - SPRING 2014
// Final project
// Victor Ceballos Inza

// This projects consists in an implementation of the marching cubes algorithm
// in three different dimensions, using the GPU to accelerate the process.

// This file contains the rendering functions.

///////////////////////////////////////////////////////////////////////////////
// Includes, system															 //
///////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "marchingCubes_cuda.cuh"

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif


///////////////////////////////////////////////////////////////////////////////
// Declarations																 //
///////////////////////////////////////////////////////////////////////////////

// Window settings
int width = 512;
int height = 512;

// Mouse controls
int mouseLeftDown = 0;
int mouseRightDown = 0;
int mouseMiddleDown = 0;
float mouseX = 0;
float mouseY = 0;
float cameraAngleX = 0.0f;
float cameraAngleY = 0.0f;
float cameraDistance = 20.0f;

// Flags to toggle drawing
int grid = 1;
int pts = 1;
int geom = 1;

// OpenGL initialization
GLvoid initCamera();
GLvoid initLights();
GLvoid initMaterial();
GLvoid initTexture();
GLvoid initColors();
GLvoid initGL();

// Rendering functions
void drawOrigin();
void drawGrid();
void drawPoints();
void drawGeom();

// VBOs
GLuint vbo[3];  // ID of VBO for vertex arrays - 0 is reserved
				// glGenBuffers() will return non-zero id if success
				// vbo[0] - grid
				// vbo[1] - points
				// vbo[2] - geometry

// Call-backs
void display();
void reshape(int w, int h);
void timer(int millisec);
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);


///////////////////////////////////////////////////////////////////////////////
// Display callback															 //
///////////////////////////////////////////////////////////////////////////////
void display()
{
	// Clear buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	// User Interaction matrices.
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glTranslatef(0.0, 0.0, -cameraDistance);	// Zooming
    glRotatef(cameraAngleX, 1.0, 0.0, 0.0);		// Rotations
    glRotatef(cameraAngleY, 0.0, 1.0, 0.0);

    // Draw grid/points/geometry
	if(grid) { drawGrid();   }
	if(pts)  { drawPoints(); }
	if(geom) { drawGeom();   }

	// Mark position of the origin
	drawOrigin();

    // User Interaction matrices.
	glPopMatrix();

	// Swap buffers.
    glutSwapBuffers();
    glutPostRedisplay();
}


///////////////////////////////////////////////////////////////////////////////
// Rendering functions														 //
///////////////////////////////////////////////////////////////////////////////
void drawOrigin()
{
	glColor3f(1.0,1.0,0.0);
	glBegin(GL_POINTS);
	glVertex4f(0.0,0.0,0.0,1.0);
	glEnd();
	glColor3f(0.0,0.0,0.0);
}

void drawGrid()
{
    // Bind VBOs with IDs.
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

	// Enable vertex arrays.
    glEnableClientState(GL_VERTEX_ARRAY);

    // Specify pointer to vertex array.
    glVertexPointer(4, GL_FLOAT, 0, 0);

    // Set up the polygon mode.
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	// Set up the color.
	glColor3f(0.0,0.0,1.0);

	// Switch on the dimension
	int n = getNumPoints();
	int size = 0;
	switch( getDimension() ) {

	case 1:
		// Get the right size of the grid to render
		size = (n-1)*2;
		// Render to screen
		glDrawArrays(GL_LINES, 0, size);
		break;

	case 2:
		// Get the right size of the grid to render
		size = (n-1)*(n-1)*4;
		// Render to screen
		glDrawArrays(GL_QUADS, 0, size);
		break;

	case 3:
		// Get the right size of the grid to render
		size = (n-1)*(n-1)*(n-1)*16;
		// Render to screen
		glDrawArrays(GL_QUADS, 0, size);
		break;
	}

	// Set color to default.
	glColor3f(0.0,0.0,0.0);

    // Disable vertex arrays.
    glDisableClientState(GL_VERTEX_ARRAY);

    // Release VBOs with ID 0 after use.
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void drawPoints()
{
    // Bind VBOs with IDs.
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);

	// Enable vertex arrays.
    glEnableClientState(GL_VERTEX_ARRAY);

    // Specify pointer to vertex array.
    glVertexPointer(4, GL_FLOAT, 0, 0);

    // Set up the polygon mode.
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	// Switch on the dimension
	int n = getNumPoints();
	int size = 0;
	switch( getDimension() ) {

	case 1:
		// Get the right number of points to render
		size = n;
		break;

	case 2:
		// Get the right number of points to render
		size = n*n;
		break;

	case 3:
		// Get the right number of points to render
		size = n*n*n;
		break;
	}

    // Render to screen
	glDrawArrays(GL_POINTS, 0, size);

    // Disable vertex arrays.
    glDisableClientState(GL_VERTEX_ARRAY);

    // Release VBOs with ID 0 after use.
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void drawGeom()
{
    // Bind VBOs with IDs.
    glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);

	// Enable vertex arrays.
    glEnableClientState(GL_VERTEX_ARRAY);

    // Specify pointer to vertex array.
    glVertexPointer(4, GL_FLOAT, 0, 0);

	// Set up the color.
	glColor3f(1.0,0.0,0.0);

	// Switch on the dimension
	int n = getNumPoints();
	int size = 0;
	switch( getDimension() ) {

	case 1:
		// Get the right dimension of the surface to render
		size = (n-1);
		// Set up the polygon mode.
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		// Render to screen
		glDrawArrays(GL_POINTS, 0, size);
		break;

	case 2:
		// Get the right dimension of the surface to render
		size = (n-1)*(n-1)*4;
		// Set up the polygon mode.
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		// Render to screen
		glDrawArrays(GL_LINES, 0, size);
		break;

	case 3:
		// Get the right dimension of the surface to render
		size = (n-1)*(n-1)*(n-1)*15;
		// Set up the polygon mode.
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		// Render to screen
		glDrawArrays(GL_TRIANGLES, 0, size);
		// Set up the color.
		glColor3f(0.0,1.0,0.0);
		// Set up the polygon mode.
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		// Render to screen
		glDrawArrays(GL_TRIANGLES, 0, size);
		break;
	}

	// Set color to default.
	glColor3f(0.0,0.0,0.0);

    // Disable vertex arrays.
    glDisableClientState(GL_VERTEX_ARRAY);

    // Release VBOs with ID 0 after use.
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


///////////////////////////////////////////////////////////////////////////////
// GLUT call-back functions													 //
///////////////////////////////////////////////////////////////////////////////

// GLUT calls this function when the windows is resized
void reshape(int w, int h)
{
	// Save new screen dimensions
	width = (GLdouble) w;
	height = (GLdouble) h;

	// Ensuring our windows is a square
    if (height == 0) { height = 1; };

    if (width > height) { width = height; }
    else { height = width; };

	// Tell OpenGL to use the whole window for drawing
	glViewport(0, 0, (GLsizei) width, (GLsizei) height);

    // Tell GLUT to call the redrawing function
    glutPostRedisplay();
}

// GLUT redraws every given milliseconds
void timer(int millisec)
{
    glutTimerFunc(millisec, timer, millisec);
    glutPostRedisplay();
}

// GLUT calls this function when a key is pressed
void keyboard(unsigned char key, int x, int y)
{

	int n = getNumPoints();

	switch(key)
	{
		// Quit when ESC or 'q' is pressed
	    case 27:
	    case 'q':
	    case 'Q':
	        exit(0);
	        break;

	    // Add a point when '+' is pressed
	    case '+':
	        n += 1;
	        break;

	    // Remove a point when '+' is pressed
	    case '-':
	        n -= 1;
	        break;

	    // Run/Stop the Marching Cubes algorithm when 'm' is pressed
	    case 'm':
	    case 'M':
	    	setCUDA();
	        break;

	    // Change the surface to render when 'f' is pressed
	    case 'f':
	    case 'F':
	    	changeFunction();
	        break;

	    // Run the 1D algorithm when '1' is pressed
	    case '1':
	    	setDimension(1);
	    	break;

	    // Run the 2D algorithm when '2' is pressed
	    case '2':
	    	setDimension(2);
	    	break;

		// Run the 3D algorithm when '3' is pressed
		case '3':
			setDimension(3);
			break;

		// Show/hide the grid when 'g' is pressed
	    case 'g':
	    case 'G':
	    	grid = 1-grid;
	    	break;

		// Show/hide the points when 'g' is pressed
	    case 'p':
	    case 'P':
	    	pts = 1-pts;
	    	break;

		// Show/hide the surface when 's' is pressed
	    case 's':
	    case 'S':
	    	geom = 1-geom;
	    	break;

	    default:
	        ;
	}

	if (n>1) { setNumPoints(n); }
	deleteVBOs(vbo);
	createVBOs(vbo);

}

// GLUT calls this function when a mouse button is pressed
void mouse(int button, int state, int x, int y)
{
    mouseX = x;
    mouseY = y;

    if(button == GLUT_LEFT_BUTTON)
    {
        if(state == GLUT_DOWN)
        {
            mouseLeftDown = 1;
        }
        else if(state == GLUT_UP)
            mouseLeftDown = 0;
    }

    else if(button == GLUT_RIGHT_BUTTON)
    {
        if(state == GLUT_DOWN)
        {
            mouseRightDown = 1;
        }
        else if(state == GLUT_UP)
            mouseRightDown = 0;
    }

    else if(button == GLUT_MIDDLE_BUTTON)
    {
        if(state == GLUT_DOWN)
        {
            mouseMiddleDown = 1;
        }
        else if(state == GLUT_UP)
            mouseMiddleDown = 0;
    }
}

// GLUT calls this function when mouse is moved while a button is held down
void motion(int x, int y)
{
    if(mouseLeftDown)
    {
        cameraAngleY += (x - mouseX);
        cameraAngleX += (y - mouseY);
        mouseX = x;
        mouseY = y;
    }
    if(mouseRightDown)
    {
        cameraDistance -= (y - mouseY) * 0.2f;
        mouseY = y;
    }
}


///////////////////////////////////////////////////////////////////////////////
// OpenGL initialization													 //
///////////////////////////////////////////////////////////////////////////////

// Sets up projection and modelview matrices.
GLvoid initCamera()
{
    // Set up the perspective matrix.
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // FOV, AspectRatio, NearClip, FarClip
    gluPerspective(60.0f, (float)(width)/height, 1.0f, 1000.0f);

    // Set up the camera matrices.
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

// Sets up OpenGL lights.
GLvoid initLights()
{
	// Define each color component.
	GLfloat ambient[]  = {0.2f, 0.2f, 0.2f, 1.0f};
	GLfloat diffuse[]  = {0.7f, 0.7f, 0.7f, 1.0f};
	GLfloat specular[] = {1.0f, 1.0f, 1.0f, 1.0f};

	// Set each color component.
	glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specular);

    // Define and set position.
    float lightPos[4] = {0, 0, 20, 1};
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

	// Turn on lighting.
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);
}

// Sets the OpenGL material state.
GLvoid initMaterial()
{
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
}

// Initialize OpenGL texture.
GLvoid initTexture()
{
	glEnable(GL_TEXTURE_2D);
}

// Sets up OpenGL colors
GLvoid initColors()
{
    glClearColor(1.0,1.0,1.0,1.0);
    glColor3f(0.0,0.0,0.0);
    glLineWidth(1.0);
    glPointSize(5.0);
}

// Sets up OpenGL state.
GLvoid initGL()
{
	// Shading method: GL_SMOOTH or GL_FLAT
	glShadeModel(GL_SMOOTH);

	// Enable depth-buffer test.
	glEnable(GL_DEPTH_TEST);

	// Set the type of depth test.
	glDepthFunc(GL_LEQUAL);

	// 0 is near, 1 is far
	glClearDepth(1.0f);

	// Set camera settings.
	initCamera();

	// Set texture settings.
	initTexture();

	// Set lighting settings.
	initLights();

	// Set material settings.
	initMaterial();

	// Set color settings.
	initColors();
}


///////////////////////////////////////////////////////////////////////////////
// Program main																 //
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	// Check whether the number of points is given as an input.
    if (argc == 2) {
    	int n = atoi(argv[1]);
    	if (n>1) { setNumPoints(n); }
    }

	// Print usage
	printf("This is an implementation of the marching cubes algorithm in CUDA.\n"
			"\nAlgorithm options:\n"
			"\t'm'         stops/starts the algorithm.\n"
			"\t'f'         changes the function of the surface to render.\n"
			"\t'1','2','3' changes the dimension of the algorithm.\n"
			"\t'+','-'     changes the number of points in the algorithm.\n"
			"\nRendering options:\n"
			"\t'g' shows/hides the grid.\n"
			"\t'p' shows/hides the points.\n"
			"\t's' shows/hides the resulting surface.\n"
			"\nViewing options:\n"
			"\t'ESC' and 'q' quits the program.\n"
			"\t'Left click' rotates the figure.\n"
			"\t'Right click' zooms in/out.\n\n");

    // Initialize GLUT
	glutInit(&argc, argv);

	// Display mode
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH | GLUT_STENCIL);

	// Initialize the window settings.
	glutInitWindowSize(width, height);
    glutInitWindowPosition(800, 200);
	glutCreateWindow("Display");

	// Initialize the scene.
	initGL();

	// Initialize the data.
	createVBOs(vbo);

	// Set up GLUT call-backs.
	glutDisplayFunc(display);
	glutTimerFunc(33, timer, 33); // redraw only every given millisec
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	// Start the GLUT main loop
	glutMainLoop();

	return 0;
}


