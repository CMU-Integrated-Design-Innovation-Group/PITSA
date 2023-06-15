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

# Python code file “RBES2G_Server.py” sets up the region-based exhaustive 
# search method for the two-dimensional problem on the GPU-based server.

from numba import cuda
import numpy as np
import math
import time
import setup_2G as setup


@cuda.jit
def check(x1_min, x1_max, x2_min, x2_max, 
          result1, result2, result3, n, n_ref):
    
    """
    
    Function "check" checks a series of undefined 2D regions, and classifies 
    each of these regions as a feasible/infeasible/indeterminate region.
    This function is designed to be performed on CUDA device
    
    x1_min, x1_max, x2_min, x2_max: cuda array
        the min or max value of x1/x2 value of each each region in the 
        series
    
    The length of these arrays is equal to the number of undefined regions to
    be checked in this function
    
    """
    limit = 0
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n:
        # Extract the corresponding min or max x1/x2 values of a region 
        # from the input arrays based on the thread index
        x11 = x1_min[idx]
        x12 = x1_max[idx]
        x21 = x2_min[idx]
        x22 = x2_max[idx]
        for i in range(n_ref**2):
            a1 = i // n_ref
            a2 = i %  n_ref
            x11_temp = x11 + a1/n_ref * (x12-x11)
            x12_temp = x11 + (a1+1)/n_ref * (x12-x11)
            x21_temp = x21 + a2/n_ref * (x22-x21)
            x22_temp = x21 + (a2+1)/n_ref * (x22-x21)
            # Use the function defined in the setup file to calculate the range 
            # of the output of the constraint function
            gg1_temp, gg2_temp = setup.g1(x11_temp, x12_temp, 
                                          x21_temp, x22_temp)
            if i == 0:
                gg1 = gg1_temp
                gg2 = gg2_temp
            else:
                gg1 = min(gg1, gg1_temp)
                gg2 = max(gg2, gg2_temp)
        # update the min value array
        result1[idx] = gg1
        # update the max value array
        result2[idx] = gg2
        # classify this region based on lower and upper limit and update the 
        # array of classification result
        # -1: infeasible region; 1: feasible region; 0: indeterminate region
        if gg1 > limit:
            result3[idx] = -1
        elif gg2 <= limit:
            result3[idx] = 1
        else:
            result3[idx] = 0

if __name__ == "__main__":
    # Input the required value of precision level (resolution)
    dx = input('The objective level of resolution for the search process = ')
    dx = float(dx)
    
    # Input the number of subintervals in each dimension for using splitting
    n_ref = input('Number of refinements = ')
    n_ref = int(n_ref)

    # Determine the number of intervals in each dimension
    n = math.ceil(4*np.pi/dx)
    print(n)

    # Number of threads per block
    tpb = 512

    # Record starting timestamp
    start = time.time()
    n_batches = math.ceil(n/1000)

    # n_batches = math.ceil(n_batches/10000) * 10000
    # Pick sample values in each dimension to generate sample points in the 
    # design space based on the input level of resolution
    xx1 = np.linspace(-2*np.pi, 2*np.pi, n+1)
    xx2 = np.linspace(-2*np.pi, 2*np.pi, n+1)
    # Determine the boundaries for data batching
    ind = np.linspace(0, n+1, n_batches+1).astype(int)

    # Counters for classified results of different regions
    count1 = 0
    count2 = 0
    count3 = 0
    for i in range(n_batches**2):
        print(f'Batch {i+1} / {n_batches**2} started')
        # Find the index in each dimension corresponding to current iteration
        i1 = i//n_batches%n_batches
        i2 = i%n_batches
        # Pick the corresponding sample intervals in each dimension
        x1 = xx1[ind[i1]:ind[i1+1]+1]
        x2 = xx2[ind[i2]:ind[i2+1]+1]
        # At this step, the expression of each 2D region is transformed from
        # its ranges on both dimensions to its min and max value on both x1 and
        # x2 dimensions. Thus, a single region is epressed using four values, 
        # which correpsonds with the required input variables for the "check"
        # function. 
        x1_min, x2_min = np.meshgrid(x1[:-1], x2[:-1])
        x1_max, x2_max = np.meshgrid(x1[1:],  x2[1:])
        # The coordinates of each 4D region are stored in the form of four 
        # values: min and max values of x1/x2 respectively
        x1_min = x1_min.flatten()
        x1_max = x1_max.flatten()
        x2_min = x2_min.flatten()
        x2_max = x2_max.flatten()
        # Transfer the coordinate values from CPU to GPU
        x1_min_gpu = cuda.to_device(x1_min)
        x1_max_gpu = cuda.to_device(x1_max)
        x2_min_gpu = cuda.to_device(x2_min)
        x2_max_gpu = cuda.to_device(x2_max)
        nn_temp = len(x1_min)
        # Create and upload array to store the classification results, and the 
        # length of each result array is equal to the number of regions
        res1 = cuda.device_array(nn_temp)
        res2 = cuda.device_array(nn_temp)
        res3 = cuda.device_array(nn_temp)
        # Parallel process the "check" function for all regions in this batch
        bpg = math.ceil(len(x1_min)**2/tpb)
        check[bpg, tpb](x1_min_gpu, x1_max_gpu, x2_min_gpu, x2_max_gpu, 
                        res1, res2, res3, nn_temp, n_ref)
        cuda.synchronize()
        # Transfer the classification result back to CPU and get the number of 
        # regions in each class
        values, counts = np.unique(res3.copy_to_host(), return_counts=True)
        if -1 in values:
            ind1 = np.where(values==-1)
            count1 += counts[ind1][0]
        if 0 in values:
            ind2 = np.where(values==0)
            count2 += counts[ind2][0]
        if 1 in values:
            ind3 = np.where(values==1)
            count3 += counts[ind3][0]
        print(f'Time spent so far: {time.time()-start}s')
    # Output the total number of regions in each class
    print(f'Number of feasible regions: {count3}')
    print(f'Number of infeasible regions: {count1}')
    print(f'Number of indeterminate regions: {count2}')
    # Report the program running time
    end = time.time()
    print(f'Total runtime: {end-start}s')