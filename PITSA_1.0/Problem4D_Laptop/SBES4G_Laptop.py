'''
Copyright 2022 Carnegie Mellon University. All rights reserved.

PITSA

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

 

YOU MAY OBTAIN A COPY OF THE AGREEMENT AT

https://github.com/CMU-Integrated-Design-Innovation-Group/PITSA

 

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF

THIS LICENSE AGREEMENT. IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY

NOT USE OR DOWNLOAD THE SOFTWARE.

 

IF YOU ARE INTERESTED IN A COMMERCIAL LICENSE, PLEASE CONTACT CMU'S CENTER

FOR TECHNOLOGY TRANSFER AND ENTERPRISE CREATION AT INNOVATION@CMU.EDU.

'''

# Python code file “SBES4G_Laptop.py” sets up the sampling-based exhaustive 
# search method for the four-dimensional problem on the GPU-based laptop.

from numba import cuda, jit
import numpy as np
import math
import time
import setup_4G_SBES as setup

# User needs to enter an interger at this step as the number of sample points 
# in each dimension for the sampling-based exhaustive search.

n = input('The objective number of sample points in each dimension = ')
n = int(n)

@cuda.jit
def check(x1, x2, x3, x4, res, n):
    """
    Function "check" checks a series of 4D sample points, and classify each of 
    these samples as a feasible/infeasible point.

    Parameters
    ----------
    x1 : cuda array
        The x1 coordinates of all the 4D sample points
    x2 : cuda array
        The x2 coordinates of all the 4D sample points
    x3 : cuda array
        The x3 coordinates of all the 4D sample points
    x4 : cuda array
        The x4 coordinates of all the 4D sample points
    res : cuda array
        This array stores the classification results of all samples.
    n : int
        Number of sample points to be checked in this function.

    Returns
    -------
    None.

    """
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n:
        # Pick the corresponding sample from input array using the thread index
        xx1 = x1[idx]
        xx2 = x2[idx]
        xx3 = x3[idx]
        xx4 = x4[idx]
        # Classify this sample point by evaluating all constraint functions at
        # this sample, and update the result array with the classification
        # result
        # 0: infeasible point; 1: feasible point
        gg1 = setup.g1(xx1, xx2, xx3, xx4)
        gg2 = setup.g2(xx1, xx2, xx3, xx4)
        gg3 = setup.g3(xx1, xx2, xx3, xx4)
        gg4 = setup.g4(xx1, xx2, xx3, xx4)
        gg5 = setup.g5(xx1, xx2, xx3, xx4)
        gg6 = setup.g6(xx1, xx2, xx3, xx4)
        gg7 = setup.g7(xx1, xx2, xx3, xx4)
        gg8 = setup.g8(xx1, xx2, xx3, xx4)
        # A sample is classified as infeasible if any of the constraint
        # functions has a positive output at this sample
        if (gg1>0 or gg2>0 or gg3>0 or gg4>0 or 
            gg5>0 or gg6>0 or gg7>0 or gg8>0):
            res[idx] = 0
        else:
            res[idx] = 1

# Number of sample values in each dimension for one batch
n_samples = 50
# Number of threads per block
tpb = 256

start = time.time()
# Use the meshgrid to pick sample points based on input number of sample points
n_batches = math.ceil((n+1)/n_samples)
xx1 = np.linspace(0.001, 1, n+1)
xx2 = np.linspace(0.001, 8, n+1)
xx3 = np.linspace(5, 30, n+1)
xx4 = np.linspace(0.001, 1, n+1)
ind = np.linspace(0, n+1, n_batches+1).astype(int)
count = [0, 0]
if __name__ == "__main__":
    # Separate all the sample points into multiple batches and perform data 
    # processing in batches
    for ii0 in range(n_batches**4):
        # Convert the current iteration number into the index in each dimension
        i1 = ii0 // n_batches**3 % n_batches
        i2 = ii0 // n_batches**2 % n_batches
        i3 = ii0 // n_batches**1 % n_batches
        i4 = ii0 // n_batches**0 % n_batches
        # Pick the corresponding sample values in each dimension
        x1 = xx1[ind[i1]:ind[i1+1]]
        x2 = xx2[ind[i2]:ind[i2+1]]
        x3 = xx3[ind[i3]:ind[i3+1]]
        x4 = xx4[ind[i4]:ind[i4+1]]
        # Transform 1D sample values to 4D coordinates using meshgrid function
        x1_mesh, x2_mesh, x3_mesh, x4_mesh = np.meshgrid(x1, x2, x3, x4)
        x1_mesh = x1_mesh.flatten()
        x2_mesh = x2_mesh.flatten()
        x3_mesh = x3_mesh.flatten()
        x4_mesh = x4_mesh.flatten()
        # Transfer the coordinates of all 4D sample points from CPU to GPU
        x1_device = cuda.to_device(x1_mesh)
        x2_device = cuda.to_device(x2_mesh)
        x3_device = cuda.to_device(x3_mesh)
        x4_device = cuda.to_device(x4_mesh)
        nn_temp = len(x1_mesh)
        # Create and upload arrays to store the classification result, and the 
        # length of this array is equal to the number of samples in this batch 
        res = cuda.device_array(nn_temp)
        # Parallel process the "check" function for all samples in this batch
        # The result array is updated with the classification result of each 
        # sample, and the update process is performed within "check" function
        bpg = math.ceil(nn_temp/tpb)
        check[bpg, tpb](x1_device, x2_device, x3_device, x4_device, 
                        res, nn_temp)
        cuda.synchronize()
        # Transfer the classification results from GPU back to CPU and get the 
        # number of samples in each class
        res = res.copy_to_host()
        values, counts = np.unique(res, return_counts=True)
        if 0 in values:
            ind0 = np.where(values==0)
            count[0] += counts[ind0][0]
        if 1 in values:
            ind0 = np.where(values==1)
            count[1] += counts[ind0][0]
        print(f'Batch {ii0+1}/{n_batches**4} finished')
        print(f'Time spent so far: {time.time()-start}s')
    # Output the classification results for all sample points
    print(f'Number of feasible solutions: {count[1]}')
    print(f'Number of infeasible solutions: {count[0]}')
# Output the program running time
end = time.time()
print(f'Total runtime: {end-start}s')
