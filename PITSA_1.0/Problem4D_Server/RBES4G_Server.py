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

# Python code file “RBES4G_Server.py” sets up the region-based exhaustive 
# search method for the four-dimensional problem on the GPU-based server.

from numba import cuda
import numpy as np
import math
import setup_4G as setup
import time


@cuda.jit
def check(x1_min, x1_max, x2_min, x2_max, 
          x3_min, x3_max, x4_min, x4_max, 
          res, n, n_ref):
    """
    
    Function "check" checks a series of undefined 4D regions, and classify each 
    of these regions as a feasible/infeasible/indeterminate region. Notably,
    each regions is denoted as its upper and lower bounds (min and max values) 
    on each of the four dimensions. 
    This function is designed to be performed on CUDA device
    
    x1_min, x1_max, x2_min, x2_max, x3_min, x3_max, x4_min, x4_max: cuda array
        the min or max value of x1/x2/x3/x4 value of each each region in the 
        series
    
    The length of these arrays is equal to the number of undefined regions to
    be checked in this function
    
    """
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n: 
        # Extract the corresponding min or max x1/x2/x3/x4 values of a region 
        # from the input arrays based on the thread index
        x11 = x1_min[idx]
        x12 = x1_max[idx]
        x21 = x2_min[idx]
        x22 = x2_max[idx]
        x31 = x3_min[idx]
        x32 = x3_max[idx]
        x41 = x4_min[idx]
        x42 = x4_max[idx]
        for i in range(n_ref**4):
            a1 = i // n_ref**3 % n_ref
            a2 = i // n_ref**2 % n_ref
            a3 = i // n_ref**1 % n_ref
            a4 = i // n_ref**0 % n_ref
            x1min = x11 + a1/n_ref * (x12-x11)
            x1max = x11 + (a1+1)/n_ref * (x12-x11)
            x2min = x21 + a2/n_ref * (x22-x21)
            x2max = x21 + (a2+1)/n_ref * (x22-x21)
            x3min = x31 + a3/n_ref * (x32-x31)
            x3max = x31 + (a3+1)/n_ref * (x32-x31)
            x4min = x41 + a4/n_ref * (x42-x41)
            x4max = x41 + (a4+1)/n_ref * (x42-x41)
            # Use the functions defined in the setup file to calculate the 
            # range of the output of all constraint functions
            g1min_temp, g1max_temp = setup.g1(x1min, x1max, x2min, x2max, 
                                    x3min, x3max, x4min, x4max)
            g2min_temp, g2max_temp = setup.g2(x1min, x1max, x2min, x2max, 
                                    x3min, x3max, x4min, x4max)
            g3min_temp, g3max_temp = setup.g3(x1min, x1max, x2min, x2max, 
                                    x3min, x3max, x4min, x4max)
            g4min_temp, g4max_temp = setup.g4(x1min, x1max, x2min, x2max, 
                                    x3min, x3max, x4min, x4max)
            g5min_temp, g5max_temp = setup.g5(x1min, x1max, x2min, x2max, 
                                    x3min, x3max, x4min, x4max)
            g6min_temp, g6max_temp = setup.g6(x1min, x1max, x2min, x2max, 
                                    x3min, x3max, x4min, x4max)
            g7min_temp, g7max_temp = setup.g7(x1min, x1max, x2min, x2max, 
                                    x3min, x3max, x4min, x4max)
            g8min_temp, g8max_temp = setup.g8(x1min, x1max, x2min, x2max, 
                                    x3min, x3max, x4min, x4max)
            if i == 0:
                g1min, g1max = g1min_temp, g1max_temp
                g2min, g2max = g2min_temp, g2max_temp
                g3min, g3max = g3min_temp, g3max_temp
                g4min, g4max = g4min_temp, g4max_temp
                g5min, g5max = g5min_temp, g5max_temp
                g6min, g6max = g6min_temp, g6max_temp
                g7min, g7max = g7min_temp, g7max_temp
                g8min, g8max = g8min_temp, g8max_temp
            else:
                g1min = min(g1min, g1min_temp)
                g1max = max(g1max, g1max_temp)
                g2min = min(g2min, g2min_temp)
                g2max = max(g2max, g2max_temp)
                g3min = min(g3min, g3min_temp)
                g3max = max(g3max, g3max_temp)
                g4min = min(g4min, g4min_temp)
                g4max = max(g4max, g4max_temp)
                g5min = min(g5min, g5min_temp)
                g5max = max(g5max, g5max_temp)
                g6min = min(g6min, g6min_temp)
                g6max = max(g6max, g6max_temp)
                g7min = min(g7min, g7min_temp)
                g7max = max(g7max, g7max_temp)
                g8min = min(g8min, g8min_temp)
                g8max = max(g8max, g8max_temp)

        # Classify this region based on the upper and lower limits of all 
        # constraint functions and objective function, and update the result 
        # array with the classification result
        # -1: infeasible region, 1: feasible region, 0: indeterminate region
        if (g1min>0 or g2min>0 or g3min>0 or g4min>0 or 
            g5min>0 or g6min>0 or g7min>0 or g8min>0):
            res[idx] = -1
        elif (g1max<=0 and g2max<=0 and g3max<=0 and g4max<=0 and 
              g5max<=0 and g6max<=0 and g7max<=0 and g8max<=0):
            res[idx] = 1
        else:
            res[idx] = 0


if __name__ == "__main__":
    # Input the required number of smallest intervals in each dimension as the 
    # precision level
    n = input('The objective number of intervals in each dimension = ')
    n = int(n)
    
    # n_ref is the number of subintervals in each dimension for using splitting
    n_ref = input("Number of refinements = ")
    n_ref = int(n_ref)
    
    # Number of threads per block
    tpb = 64
    
    # Number of sample values in each dimension for one batch
    n_samples = 10
    
    # Record starting timestamp
    ts0 = time.time()
    
    # Number of batches in each dimension
    n_batches = math.ceil(n/n_samples)
    
    # Pick sample points in each dimension according to the number of intervals
    xx1 = np.linspace(0.001, 1, n+1)
    xx2 = np.linspace(0.001, 8, n+1)
    xx3 = np.linspace(5, 30, n+1)
    xx4 = np.linspace(0.001, 1, n+1)
    
    # Determine the boundaries for data batching
    ind = np.linspace(0, n+1, n_batches+1).astype(int)
    
    # Counters for classified results of different regions
    count = [0, 0, 0]
    
    for i in range(n_batches**4):
        print(f'Batch {i+1}/{n_batches**4} started')
        # Convert the current iteration number into the index in each dimension
        i1 = i//n_batches**3%n_batches
        i2 = i//n_batches**2%n_batches
        i3 = i//n_batches**1%n_batches
        i4 = i//n_batches**0%n_batches
        # Pick the corresponding sample intervals in each dimension
        x1 = xx1[ind[i1]:ind[i1+1]+1]
        x2 = xx2[ind[i2]:ind[i2+1]+1]
        x3 = xx3[ind[i3]:ind[i3+1]+1]
        x4 = xx4[ind[i4]:ind[i4+1]+1]
        # Transform 1D sample intervals to 4D coordinates that correspond to 
        # 4D sample regions using meshgrid function
        x1min, x2min, x3min, x4min = \
            np.meshgrid(x1[:-1], x2[:-1], x3[:-1], x4[:-1])
        x1max, x2max, x3max, x4max = \
            np.meshgrid(x1[1:], x2[1:], x3[1:], x4[1:])
        # The coordinates of each 4D region are stored in the form of eight 
        # values: min and max values of x1/x2/x3/x4 respectively
        x1min = x1min.flatten()
        x1max = x1max.flatten()
        x2min = x2min.flatten()
        x2max = x2max.flatten()
        x3min = x3min.flatten()
        x3max = x3max.flatten()
        x4min = x4min.flatten()
        x4max = x4max.flatten()
        # Transfer the coordinate values from CPU to GPU
        x1min_gpu = cuda.to_device(x1min)
        x1max_gpu = cuda.to_device(x1max)
        x2min_gpu = cuda.to_device(x2min)
        x2max_gpu = cuda.to_device(x2max)
        x3min_gpu = cuda.to_device(x3min)
        x3max_gpu = cuda.to_device(x3max)
        x4min_gpu = cuda.to_device(x4min)
        x4max_gpu = cuda.to_device(x4max)
        # Create and upload array to store the classification results, and the 
        # length of the result array is equal to the number of regions
        res = cuda.device_array(len(x1min))
        # Parallel process the "check" function for all regions in this batch
        bpg = math.ceil(len(x1min)/tpb)
        check[bpg, tpb](x1min_gpu, x1max_gpu, x2min_gpu, x2max_gpu, 
                        x3min_gpu, x3max_gpu, x4min_gpu, x4max_gpu, 
                        res, len(x1min), n_ref)
        cuda.synchronize()
        res = res.copy_to_host()
        # Transfer the classification result back to CPU and get the number of 
        # regions in each class
        values, counts = np.unique(res, return_counts = True)
        if -1 in values:
            ind0 = np.where(values==-1)
            count[0] += counts[ind0][0]
        if 0 in values:
            ind0 = np.where(values==0)
            count[1] += counts[ind0][0]
        if 1 in values:
            ind0 = np.where(values==1)
            count[2] += counts[ind0][0]
        print(f'Batch {i+1}/{n_batches**4} finished')
        print(f'Time spent so far: {time.time()-ts0}s')
    ts3 = time.time()
    # Output the total number of regions in each class
    print(f'Number of feasible regions: {count[2]}')
    print(f'Number of infeasible regions: {count[0]}')
    print(f'Number of indeterminate regions: {count[1]}')
    # Report the program running time
    print(f'Total runtime: {ts3-ts0}s')
