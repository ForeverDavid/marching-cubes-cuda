# A Marching Cubes implementation in CUDA

## Description

This project is an implementation of the marching cubes
algorithm in 3D, 2D (marching squares), and 1D.
The marching cubes algorithm is a simple algorithm for
creating a triangle mesh from an implicit function.
The process works as follows:

- Divide the space into an arbitrary number of cubes.
- Test the corners of every cube for whether they are
inside the object defined by the function.
- For every cube where some corners are inside and some
corners are outside the object, the surface must pass 
through that cube, intersecting the edges of the cube 
in between corners of opposite classification.
- Draw a surface within each cube connecting these 
intersections.


## Running the application

Compile the program using the makefile provided:
```
make
```

Run "marchingCubes" with either of the commands:
```
./marchingCubes
./marchingCubes N
``` 
where N is the number of points used in the algorithm.
If N is not provided, it is default to 3.


## Using the application

When marchingCubes is run, it will render to the screen a
surface defined by a built in function.
The program provides a wide user interface:

 - Algorithm options:
	'm'     stops/starts the algorithm.
	'f'     changes the function of the surface to render.
	'1'-'3' changes the dimension of the algorithm.
	'+','-' changes the number of points in the algorithm.
	
 - Rendering options:
	'g' shows/hides the grid.
	'p' shows/hides the points.
	's' shows/hides the resulting surface.
	
 - Viewing options:
	'ESC' and 'q' quits the program.
	'Left click' rotates the figure.
	'Right click' zooms in/out.


There are three built-in function per dimension. These
are:
 - 1D: 1-dim sphere, a semi-line, two intervals.
 - 2D: 2-dim sphere, a collage of geometric forms, the
 	   body of a fish.
 - 3D: 3-dim sphere, a one-sheeted hyperboloid, a 
       hyperbolic paraboloid. 

Remark: For a nicer rendering, make sure to increase the
number of points significantly, as well as turning off
the grid and zooming out.


## FAQs

1. Why does GPU help here?

	Because instead of running a version of the algorithm
	where each point is tested for containment in the
	surface sequentially, using the GPU we do this test
	in parallel. This results in a significant speed-up.
	This is an example of a parallelizable problem since
	we have a large number of copies a simple structure -
	either an interval (1D), square (2D) or cube (3D).


2. What work does one thread do per kernel call? 

	There are two kernels in this program:
	- The first one handles the containment test. Each
	thread takes care of one point in the algorithm,
	and tests it for whether it is inside the object
	defined by the function.
	- The second kernel handles the connection between
	intersections. Each thread takes care of one 
	structure (interval, square or cube) in the grid
	and connects the intersection points with the
	object defined by the function.


3. What sorts of considerations did you make regarding memory? 

	This program uses mainly global and register memory
	in its current implementation. I considered using
	shared memory, but each threads need access to a 
	particular cube in the grid, defined by eight vertices.
	While it is true that adjacent threads need access to
	four common vertices (among others), all vertices in a
	block only share one vertex.
	This made redundant the use of shared memory.
	
	
