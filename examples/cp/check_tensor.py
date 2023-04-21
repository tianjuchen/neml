#!/usr/bin/env python3

import sys
sys.path.append('../..')

import numpy as np

from neml.cp import polycrystal, crystallography, slipharden, sliprules, inelasticity, kinematics, singlecrystal, polefigures
from neml.math import rotations, tensors, nemlmath, matrix
from neml import elasticity

import matplotlib.pyplot as plt

if __name__ == "__main__":


    L = crystallography.CubicLattice(1.0)
    L.add_slip_system([1,1,0],[1,1,1])
    
    nslip = L.ntotal
    
    M = matrix.SquareMatrix(nslip, type = "block", 
        data = [0.1,0.2,0.3,0.4], blocks = [6,6])
        
    print(np.array([M]))