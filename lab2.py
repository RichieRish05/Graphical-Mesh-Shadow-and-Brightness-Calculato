#!/usr/bin/env python
# coding: utf-8

# # Lab 2
# ___
# ## To submit
# Go to the file tab, and download this notebook as .py. Make sure this file is named lab2.py, and submit it to Gradescope.
# ___
# 1. In computer graphics, we represent physical objects in form of connected triangles. Each triangle represent part of surface of the object. Each triangle has 3 vertices, that each vertex is a point in 3D space. In fact, triangle edges connect these 3D points in a way that the triangles look like the  surface of the object. For example a simple cube can be created by 8 vertices and 12 triangles (2 triangle per side). We can create very approximate very complicated surfaces with triangle mesh (like the terrain).
# 
# <img src="images/cube_triangle.png">
# 
# <img src="images/terrain_triangle.jpg">
# 
# We know that in 3-dim space we can represent a point with a 3-dim vector. That  vector starts from the origin (0,0,0) and ends at that point. So, each triangle in a mesh is basically composed of three vectors.
# 
# 2. A simple way to store a triangular mesh in computer is keeping 2 matrices: one for vertices, and another one for triangles. Here we represent the above 2D triangular mesh with two matrices: vertex matrix (V) and triangles matrix (F). The vertex matrix (V) is an $n \times 3$ matrix that represents $n$ vertices that each vertex is a row of this matrix which is a point in 3D space. The triangle matrix (F) is an $m \times 3$ matrix that represents $m$ triangles that each triangle is a row of this matrix which is the index of vertices that form a triangle. For example the V and F matrices for this mesh is:
# 
# <img src="images/simple_mesh.jpg">
# 

# In[13]:


import numpy as np


# In[14]:


V = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0]])


F = np.array([
    [0, 1, 2],
    [0, 2, 3]])


# In the above example we have 4 vertices ($n = 4$) and 2 triangle ($m = 2$). Triangle 0 (row 0 of matrix F) is made of vertices 0, 1, and 2. Note that these numbers are the order of this triangle's vertices in the matrix V. It is customary that to represent each tringle with its vertices in counter-clockwise order. That is if we replce the first row of matrix F with 1, 2, 0 or 2, 0, 1, they all are correct way of representing triangle 0.
# 
# 3. We know cross product of two vectors is a vector perpendicular to both vectors and its size is proportional to the size of the two vectors and $sin$ of the angle between them:
# 
# $a \times b = \left\lVert a \right\rVert \left\lVert b \right\rVert \sin(\theta)$
# 
# The direction of $a \times b$ follows the right-hand rule. That is if you try to move the four fingers of your right hand from $a$ to $b$, then your thumb shows the direction of $a \times b$ as shown below.
# 
# <img src="images/right_hand_cross.png">
# 
# Obviously if you change the order of $a$ and $b$, the result $b \times a$ will be in the opposite direction of $a \times b$. Now let's look back in a sample triangle from a mesh. Suppose a triangle is made of three vertices, $v_0, v_1,$ and $v_2$ in counter clock-wise order. And let's name the three edges as $e_{0,1}, e_{1,2},$ and $e_{2,0}$, where $e_{i,j}$ is an  edge starting from $v_i$ and ending at $v_j$. Following shows a sample triangle with its vertices and edges.
# 
# <img src="images/edges_triangle.jpg">
# 
# Now try applying the right-hand rule on two consecutive edges (say $e_{0,1} \times e_{1,2}$ on your monitor). Your will see that your thumb will direct from your monitor to yourself. This direction is the direction of normal vector of this triangle. Using this method you can calculate the normal vector for each triangle in a triangular mesh. Since the normal vector of a triangle should be independent of the area of triangle, we always make the normal vector to be a unit vector. 

# ## Exercise 1
# You should complete the following python function named "calc_normal" that takes vertices (V) and triangles (F) of an input triangular mesh and return a matrix that its i-th row is the normal vector of i-th triangle in matrix F. Note that the normal vector should be a unit vector. 
# 

# In[26]:


import math

def calc_normal(V: np.array, F: np.array) -> np.array:
    
    m,_ = F.shape
    N = np.zeros((m, 3))
    
    for i in range(m):
        # calculate N[i,:]
        row_one = V[F[i,0], :] - V[F[i,1], :]
        row_two = V[F[i,0], :] - V[F[i,2], :]
        N[i, :] = np.cross(row_one, row_two)


    
    return N

print(calc_normal(V, F))


# 4. In computer graphics, creating images (frames) from a scene (objects and light sources) is called "Rendering". We see objects since they reflect light to our eyes. The light ray reaches the object surface from a light source and makes the surface visible. One of the form of light sources is a point light source. That is we assume there is a light source somewhere in 3D space, represented as a 3-dim vector, and it shines the rays of light to all directions. 
# 
# <img src="images/point light-1.png">
# 
# For matte materials the brightness of a surface is proportional to the angle between the normal direction of the surface to the light ray from the light source. That means the surface is brightest if the light ray is in the same direction of surface normal. In this assignment, for sake of simplicity, we find a single ray from each triangle to the light source. That single ray is a vector from the center of the triangle $(b)$ to the light source $(s)$.
# 
# <img src="images/light_rendering.jpg">
# 
# The center of a triangle (barycentric point) can easily be calculated as the center of mass of a triangle. We don't want to go into details on this, but you can simply calculate the center of each triangle using the following formula:
# 
# $b = \frac{1}{3} (v_0 + v_1 + v_2)$
# 
# It is a linear combination of three vertices of the triangle. Once you calculate the center of a triangle, you can then calculate the vector that starts from that point to the light source as:
# 
# $l = s - b$
# 
# The dot product between the two vectors $\hat{n}$ (unit vector of normal of triangle) and $\hat{l}$ (unit vector from triangle center to light source) is equal to cos of the angle between them. We know $\cos(\theta)$ has a value between -1 and 1. It is 1, when the angle is 0 degrees, and it is 0 when the angle is 90 degrees. That makes sense as you can imagine a surface is not lit if a point light source is parallel to the surface.
# 
# The interesting part is when $\cos(\theta)$ is negative the angle is greater than 90 degrees. It means the light ray is being emitted to the back side of the triangle (for example the light is in the other side of cube, so this side doesn't get any light and is dark).
# 
# In this exercise you should calculate the brightness of each triangle in mesh given a light source coordinate. The brightness is:
# 
# $brightness = \hat{n} \cdot \hat{l}$
# 
# if the angle between $\hat{n}$ and $\hat{l}$ is less than 90 degrees. The brightness should be zero if that angle is greater than 90 degrees.

# ## Exercise 2 
# You should complete the python function named "calc_brightness" that takes the two matrices of vertices (V) and triangles (F) which together represent a triangular mesh along with a 3D point that represent the location of light source and return a vector (numpy array), that its i-th element in output vector is the brightness of the i-th triangle in matrix F. Note that the brightness of each triangle should be an scalar between 0 to 1.
# 
# 

# In[21]:


def calc_brightness(V: np.array, F: np.array, s: np.array) -> np.array:
    
    # Finding normal for all triangles
    N = calc_normal(V,F)
    
    m,_ = F.shape
    brightness = np.zeros(m)
    for i in range(m):
        b = (V[F[i,0],:] + V[F[i,1],:] + V[F[i,1],:] + V[F[i,2],:])/3
        l = s - b
        # Find brightness[i]
        brightness[i] = np.dot(N[i, :], l)
        
    return brightness



# 5. In this lab assignment you wrote functions to perform very basic rendering calculations using two simple but fundamental operations in linear algebra (dot product and cross product of vectors). Your function calculates the brightness of each tringle depending on how it is lit by a point light source. For example you can see how the brightness of each triangle is rendered differently for the following image.
# 
# <img src="images/point_source_rendered_mesh.png">
# 
# In many computer graphics applications (e.g. computer games), such operations should take place for thousands of triangles per frame. Unlike your Python functions that perform these operations on CPU, computer graphic applications utilize the specific hardware on computer called GPU, performing all these calculations on triangles and vertices in parallel to increase the fps (frames per second). 

# In[ ]:




