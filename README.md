A lightweight GPU accelerated (cuBlas used) Basel Morphable face model generator.

## Overview

This project implements a 3D Basel face generator. 
It allows to set shape and texture using 2 vectors (199 values for shape and 199 values for texture).

On my machine mesh generation takes about 4-5 ms. 

## Prerequisites

-  Windows 10
-  MSVS 2019 
-  CUDA 10.1

- Packages   
1. [CMake](https://cmake.org/download/)

## Model preprocessing ##

You need to download matlab Basel face model.
After that, you'll need to generate npz model,
using python script data_transform.py.
Just run it from console 'python data_transform.py'.

![head](BaselHead.gif)
