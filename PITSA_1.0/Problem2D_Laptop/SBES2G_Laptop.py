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

# Python code file “SBES2G_Laptop.py” sets up the sampling-based exhaustive 
# search method for the two-dimensional problem on the GPU-based laptop.

from numba import cuda, jit
import numpy as np
import math
import time

# Number of CUDA cores available on the GPU-based device
n_cores = 756

# At this step, user needs to input a float number as the obejctive level of
# resolution for the sampling-based exhaustive search process
dx = input('The objective level of resolution for the search process = ')
dx = float(dx)
n = math.ceil(4*np.pi/dx)

@cuda.jit
def check(x1, x2, res, n):
    """
    Function "check" tests a series of 2D sample points, and classifies each of 
    these points as a feasible/infeasible point.

    Parameters
    ----------
    x1 : cuda array
        The x1 coordinates of all the 2D sample points
    x2 : cuda array
        The x2 coordinates of all the 2D sample points
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
        # Pick the corresponing values from input array using the thread index
        xx1 = x1[idx]
        xx2 = x2[idx]
        flag = 1
        # Check whether the point is a solution point
        gg1 = g1(xx1, xx2)
        if gg1 > 0:
            flag = 0
        # Update the result array with the classification result
        res[idx] = flag

@cuda.jit
def g1(x1, x2):
    # Function "g1" defines the constraint function of the 2D problem
    ans = (x1-x2)*(x1-x2) + math.sin(x1)*math.exp((1-math.cos(x2))**2) + \
        math.cos(x2)*math.exp((1-math.sin(x1))**2)
    return ans

# Number of sample values in each dimension for one batch
n_samples = 20
# Number of threads per block
tpb = 256

start = time.time()

# Use meshgrid to pick sample points based on required precision level
n_batches = math.ceil((n+1)/n_samples)
n_batches = math.ceil(n_batches/10) * 10
xx1 = np.linspace(-2*np.pi, 2*np.pi, n+1)
xx2 = np.linspace(-2*np.pi, 2*np.pi, n+1)
ind = np.linspace(0, n+1, n_batches+1).astype(int)
count = [0, 0]

if __name__ == "__main__":
    # Separate all the sample points into multiple batches and perform data 
    # processing in batches
    for ii0 in range(n_batches**2):
        # Find the index in each dimension corresponding to current iteration
        i1 = ii0 // n_batches**1 % n_batches
        i2 = ii0 // n_batches**0 % n_batches
        # Pick the corresponding sample values in each dimension
        x1 = xx1[ind[i1]:ind[i1+1]]
        x2 = xx2[ind[i2]:ind[i2+1]]
        # Transform the sample values in both dimensions to 2D coordinates 
        # using meshgrid function
        x1_mesh, x2_mesh = np.meshgrid(x1, x2)        
        x1_mesh = x1_mesh.flatten()
        x2_mesh = x2_mesh.flatten()
        # Transfer the coordinates of all 2D sample points from CPU to GPU
        x1_device = cuda.to_device(x1_mesh)
        x2_device = cuda.to_device(x2_mesh)
        nn_temp = len(x1_mesh)
        # Create and upload arrays to store the classification result, so the 
        # length of res is equal to the number of samples in this batch 
        res = cuda.device_array(nn_temp)
        # print('parallel computing')
        # Parallel process the "check" function for all samples in this batch 
        bpg = math.ceil(n_cores/tpb)
        check[bpg, tpb](x1_device, x2_device, res, nn_temp)
        cuda.synchronize()
        # Transfer the classification results from GPU back to CPU and count 
        # the number of samples in each class for the current batch
        res = res.copy_to_host()
        values, counts = np.unique(res, return_counts=True)
        if 0 in values:
            ind0 = np.where(values==0)
            count[0] += counts[ind0][0]
        if 1 in values:
            ind0 = np.where(values==1)
            count[1] += counts[ind0][0]
        if ii0 % 100 == 0:
            print(f'Batch {ii0+1}/{n_batches**2} finished')
            print(f'Time spent so far: {time.time()-start}s')

    # Output the classfication results for all sample points
    print(f'Number of infeasible solution: {count[0]}')
    print(f'Number of feasible solution: {count[1]}')

# Output the program running time
end = time.time()
print(f'Total runtime: {end-start}s')
