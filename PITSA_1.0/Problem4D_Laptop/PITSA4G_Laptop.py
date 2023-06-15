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

# Python code file “PITSA2G_Laptop.py” sets up the PITSA method for the 
# four-dimensional problem on the GPU-based laptop.

from numba import cuda
import numpy as np
import math
import time
import multiprocessing
from multiprocessing import Process
import setup_4G as setup
import csv


def mesh1(x1min, x1max, x2min, x2max, x3min, x3max, x4min, x4max, n):
    """
    Function "mesh1" partitions a 4D region into n*n small 4D regions using the 
    meshgrid function in python.

    Parameters
    ----------
    x1min/x1max/x2min/x2max/x3min/x3max/x4min/x4max: float
        The max or min x1/x2/x3/x4 values of this 4D region
    n : integer
        The number of partitioned intervals in each dimension.

    Returns
    -------
    x1_min/x1_max/x2_min/x2_max/x3_min/x3_max/x4_min/x4_max: 1D array
        Array of max or min x1/x2/x3/x4 values of all partitioned 4D regions.
        Length of these arrays is equal to the number of new regions obtained 
        from partitioning. 

    """
    x1 = np.linspace(x1min, x1max, n+1)
    x2 = np.linspace(x2min, x2max, n+1)
    x3 = np.linspace(x3min, x3max, n+1)
    x4 = np.linspace(x4min, x4max, n+1)
    x_min = np.meshgrid(x1[:-1], x2[:-1], x3[:-1], x4[:-1])
    x_max = np.meshgrid(x1[1:], x2[1:], x3[1:], x4[1:])
    # Flatten all mashed points an return as function outputs
    x1_min = x_min[0].flatten()
    x1_max = x_max[0].flatten()
    x2_min = x_min[1].flatten()
    x2_max = x_max[1].flatten()
    x3_min = x_min[2].flatten()
    x3_max = x_max[2].flatten()
    x4_min = x_min[3].flatten()
    x4_max = x_max[3].flatten()
    return x1_min, x1_max, x2_min, x2_max, x3_min, x3_max, x4_min, x4_max


def mesh2(list0, q, n):
    """
    This function performs mesh1 function repetitively for a series of 4D 
    regions.

    Parameters
    ----------
    list0 : list
        A list of undefined regions that need to be checked in this function.
    q : multiprocessing.Queue()
        This queue receives the output results from different threads in 
        python multiprocessing.
    n : int
        n is the input for mesh1 function. See mesh1 function comments for 
        detailed description.

    Returns
    -------
    None.

    """
    pts = [[], [], [], [], [], [], [], []]
    for i in range(len(list0)):
        ans = mesh1(list0[i][0], list0[i][1], list0[i][2], list0[i][3], 
                    list0[i][4], list0[i][5], list0[i][6], list0[i][7], n)
        pts[0] += list(ans[0])
        pts[1] += list(ans[1])
        pts[2] += list(ans[2])
        pts[3] += list(ans[3])
        pts[4] += list(ans[4])
        pts[5] += list(ans[5])
        pts[6] += list(ans[6])
        pts[7] += list(ans[7])
    q.put(pts)


@cuda.jit
def check(x1_min, x1_max, x2_min, x2_max, 
          x3_min, x3_max, x4_min, x4_max, 
          res, n, n_ref):
    """
    
    Function "check" checks a series of undefined 4D regions, and classify each 
    of these regions as a feasible/infeasible/indeterminate region.
    This function is designed to be performed on CUDA device
    
    x1_min, x1_max, x2_min, x2_max, x3_min, x3_max, x4_min, x4_max: cuda array
        the min or max value of x1/x2/x3/x4 value of each each region in the 
        series
    
    The length of these arrays is equal to the number of undefined regions to
    be checked in this function
    
    """
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx<n:
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
        # Apply refinement analysis in region-based analysis for a 4D region
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
            # Use the function defined in the setup file to calculate the range
            # of the output of all constraint functions
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
        # constraint functions, and update the result array with the 
        # classification result
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
    # Set a required stopping criteria by inputting a integer
    # nmax is the objective number of smallest intervals in each of the four 
    # dimensions. 
    # The algorithm stops when the number of smallest intervals in each of the 
    # four dimensions all exceeded this value
    nmax = input('The objective number of intervals in each dimension = ')
    nmax = int(nmax)

    # n_ref is the number of subintervals in each dimension for using splitting
    n_ref = input('Number of refinement = ')
    n_ref = int(n_ref)

    # Number of CUDA cores available on current device
    n_cores = 756

    # Number of threads per block
    tpb = 256

    # Record starting timestamp
    start = time.time()

    # Number of smallest intervals in each dimension, updated in each iteration
    n_interval = 1

    # Number of CPU parallel threads used in the data modification process
    n_proc = 50

    # The threshold value for maximum number of undefined regions that can be 
    # analyzed in one iteration
    n_breakout = 5000

    # The threshold value used to determine whether CPU parallel processign 
    # needs to be applied in data modfication process
    n_threshold = 5000
    output = [[], [[0.001, 1, 0.001, 8, 5, 30, 0.001, 1]], []]
    counts = [0, 0, 0]
    count_reg = 0

    # Number of iterations in this algorithm (updated in each iteration)
    ii = 0
    flag = 0
    while n_interval < nmax:
        # list0 is a list of all the indeterminate regions obtained from last 
        # iteration, and these regionsneed to be partitioned and analyzed in 
        # the current iteration
        list0 = output[1]
        ts0 = time.time()
        # Use the number of threads per block to set the partition strategy in 
        # the current iteration
        # n0 is the umber of smallest intervals that one region is partitioned 
        # into in each dimensionin the current iteration
        n0 = math.floor((n_cores / len(list0))**0.25)
        n0 = max(n0, 2)
        # Jump out of the current loop and go to the next loop for analysis in 
        # batches if there are too many regions to be analyzed
        if len(list0) > n_breakout:
            flag = 1
            break
        # Update the iteration number
        ii += 1
        # Update the nunmber of smallest intervals in each dimension
        n_interval *= n0
        # Clear the output list containing indeterminate regions
        output[1] = []
        # Update the region counters
        counts[0] *= n0**4
        counts[1] = 0
        counts[2] *= n0**4
        # The number of partitioned regions to be analyzed in this iteration
        nn = len(list0) * n0**4
        count_reg += nn
        # This part modifies all indeterminate regions into the form that can 
        # be processed with the "check" function defined above
        # If the number of undefined regions exceed a certain threshold, aply 
        # CPU parallel processing in this process
        if len(list0) >= n_threshold:
            # Partition the original list of indeterminate regions based on the
            # number of parallel threads
            ind = np.linspace(0, len(list0), n_proc+1).astype(int)
            list0_div = [list0[ind[i]:ind[i+1]] for i in range(n_proc)]
            # Create parallel processes and assign corresponding regions to 
            # each process
            q = multiprocessing.Queue()
            jobs = []
            for i in range(n_proc):
                p = Process(target=mesh2, args=(list0_div[i], q, n0))
                jobs.append(p)
            for p in jobs:
                p.start()
            for p in jobs:
                p.join(timeout=1e-8)
            # Extract and modify the output results
            results = [q.get() for j in jobs]
            x1_min = []
            x1_max = []
            x2_min = []
            x2_max = []
            x3_min = []
            x3_max = []
            x4_min = []
            x4_max = []
            for res in results:
                x1_min += res[0]
                x1_max += res[1]
                x2_min += res[2]
                x2_max += res[3]
                x3_min += res[4]
                x3_max += res[5]
                x4_min += res[6]
                x4_max += res[7]
        # Data modification with serial processing
        else:
            x1_temp = [np.linspace(var[0], var[1], n0+1) for var in list0]
            x2_temp = [np.linspace(var[2], var[3], n0+1) for var in list0]
            x3_temp = [np.linspace(var[4], var[5], n0+1) for var in list0]
            x4_temp = [np.linspace(var[6], var[7], n0+1) for var in list0]
            xx_min = [np.meshgrid(x1_temp[i][:-1], x2_temp[i][:-1], 
                                  x3_temp[i][:-1], x4_temp[i][:-1]) 
                      for i in range(len(list0))]
            xx_max = [np.meshgrid(x1_temp[i][1:], x2_temp[i][1:], 
                                  x3_temp[i][1:], x4_temp[i][1:]) 
                      for i in range(len(list0))]
            x1_min = np.hstack([xx_min[i][0].flatten() 
                                for i in range(len(list0))])
            x1_max = np.hstack([xx_max[i][0].flatten() 
                                for i in range(len(list0))])
            x2_min = np.hstack([xx_min[i][1].flatten() 
                                for i in range(len(list0))])
            x2_max = np.hstack([xx_max[i][1].flatten() 
                                for i in range(len(list0))])
            x3_min = np.hstack([xx_min[i][2].flatten() 
                                for i in range(len(list0))])
            x3_max = np.hstack([xx_max[i][2].flatten() 
                                for i in range(len(list0))])
            x4_min = np.hstack([xx_min[i][3].flatten() 
                                for i in range(len(list0))])
            x4_max = np.hstack([xx_max[i][3].flatten() 
                                for i in range(len(list0))])
        # Create multiple empty arrays as result arrays for label collection
        res = cuda.device_array(nn)
        # Send all undefined region data to GPU
        x1_min_gpu = cuda.to_device(np.array(x1_min))
        x1_max_gpu = cuda.to_device(np.array(x1_max))
        x2_min_gpu = cuda.to_device(np.array(x2_min))
        x2_max_gpu = cuda.to_device(np.array(x2_max))
        x3_min_gpu = cuda.to_device(np.array(x3_min))
        x3_max_gpu = cuda.to_device(np.array(x3_max))
        x4_min_gpu = cuda.to_device(np.array(x4_min))
        x4_max_gpu = cuda.to_device(np.array(x4_max))
        ts1 = time.time()
        # number of blocks in the grid
        bpg = math.ceil(len(x1_min)/tpb)
        # Call the "check" function to process all the undefined regions
        check[bpg, tpb](
            x1_min_gpu, x1_max_gpu, x2_min_gpu, x2_max_gpu, 
            x3_min_gpu, x3_max_gpu, x4_min_gpu, x4_max_gpu, 
            res, nn, n_ref)
        # Synchronize all the threads
        cuda.synchronize()
        ts2 = time.time()
        # Copy the list containing all labels from GPU back to CPU
        res = res.copy_to_host()
        # Count the number of partitioned regions in each class
        values, counts_label = np.unique(res, return_counts=True)
        if -1 in values:
            ind = np.where(values==-1)
            counts[0] += counts_label[ind][0]
        if 0 in values:
            ind = np.where(values==0)
            counts[1] += counts_label[ind][0]
        if 1 in values:
            ind = np.where(values==1)
            counts[2] += counts_label[ind][0]
            
        # Collect and classify all partitioned regions in current iteration 
        # according to the results of the interval evaluations. 
        temp1 = [[x1_min[i], x1_max[i], x2_min[i], x2_max[i], 
                  x3_min[i], x3_max[i], x4_min[i], x4_max[i]] 
                 for i in range(nn) if res[i] == -1]
        temp2 = [[x1_min[i], x1_max[i], x2_min[i], x2_max[i], 
                  x3_min[i], x3_max[i], x4_min[i], x4_max[i]] 
                 for i in range(nn) if res[i] == 0]
        temp3 = [[x1_min[i], x1_max[i], x2_min[i], x2_max[i], 
                  x3_min[i], x3_max[i], x4_min[i], x4_max[i]] 
                 for i in range(nn) if res[i] == 1]
        # Update the output lists with new partitioned regions in each class
        output[0].extend(temp1)
        output[1].extend(temp2)
        output[2].extend(temp3)
        count = sum(counts)
        ts3 = time.time()
        # Print basic information after each iteration
        print(f'Iteration {ii}: ')
        print(f'\t Number of smallest intervals (one dimension): {n_interval}')
        print('results:')
        print(f'\t feasible regions:      {counts[2]/count*100} %')
        print(f'\t infeasible regions:    {counts[0]/count*100} %')
        print(f'\t indeterminate regions: {counts[1]/count*100} %')

    while flag == 1:
        if n_interval >= nmax:
            break
        list0_tot = output[1]
        ts0 = time.time()
        n0 = math.floor((n_cores / len(list0_tot))**0.25)
        n0 = max(n0, 2)
        ii += 1
        n_interval *= n0

        output[1] = []
        counts[0] *= n0**4
        counts[2] *= n0**4
        counts[1] = 0
        n_iter = math.ceil(len(list0_tot) / n_breakout)
        list_ind = np.linspace(0, len(list0_tot), n_iter+1).astype(int)
        for i_iter in range(n_iter):
            ts0 = time.time()
            list0 = list0_tot[list_ind[i_iter]:list_ind[i_iter+1]]
            nn = len(list0) * n0**4
            count_reg += nn
            n_proc = 50
            ind = np.linspace(0, len(list0), n_proc+1).astype(int)
            list0_div = [list0[ind[i]:ind[i+1]] for i in range(n_proc)]
            q = multiprocessing.Queue()
            jobs = []
            for i in range(n_proc):
                p = Process(target=mesh2, args=(list0_div[i], q, n0))
                jobs.append(p)
            for p in jobs:
                p.start()
            for p in jobs:
                p.join(timeout=1e-8)
            results = [q.get() for j in jobs]
            x1_min = []
            x1_max = []
            x2_min = []
            x2_max = []
            x3_min = []
            x3_max = []
            x4_min = []
            x4_max = []
            for res in results:
                x1_min += res[0]
                x1_max += res[1]
                x2_min += res[2]
                x2_max += res[3]
                x3_min += res[4]
                x3_max += res[5]
                x4_min += res[6]
                x4_max += res[7]
            res = cuda.device_array(nn)
            x1_min_gpu = cuda.to_device(np.array(x1_min))
            x1_max_gpu = cuda.to_device(np.array(x1_max))
            x2_min_gpu = cuda.to_device(np.array(x2_min))
            x2_max_gpu = cuda.to_device(np.array(x2_max))
            x3_min_gpu = cuda.to_device(np.array(x3_min))
            x3_max_gpu = cuda.to_device(np.array(x3_max))
            x4_min_gpu = cuda.to_device(np.array(x4_min))
            x4_max_gpu = cuda.to_device(np.array(x4_max))
            ts1 = time.time()
            # number of blocks in the grid
            bpg = math.ceil(len(x1_min)/tpb)
            check[bpg, tpb](
                x1_min_gpu, x1_max_gpu, x2_min_gpu, x2_max_gpu, 
                x3_min_gpu, x3_max_gpu, x4_min_gpu, x4_max_gpu, res, 
                nn, n_ref)
            cuda.synchronize()
            ts2 = time.time()
            res = res.copy_to_host()
            values, counts_label = np.unique(res, return_counts=True)
            if -1 in values:
                ind = np.where(values==-1)
                counts[0] += counts_label[ind][0]
            if 0 in values:
                ind = np.where(values==0)
                counts[1] += counts_label[ind][0]
            if 1 in values:
                ind = np.where(values==1)
                counts[2] += counts_label[ind][0]
            temp1 = [[x1_min[i], x1_max[i], x2_min[i], x2_max[i], 
                      x3_min[i], x3_max[i], x4_min[i], x4_max[i]] 
                     for i in range(nn) if res[i] == -1]
            temp2 = [[x1_min[i], x1_max[i], x2_min[i], x2_max[i], 
                      x3_min[i], x3_max[i], x4_min[i], x4_max[i]] 
                     for i in range(nn) if res[i] == 0]
            temp3 = [[x1_min[i], x1_max[i], x2_min[i], x2_max[i], 
                      x3_min[i], x3_max[i], x4_min[i], x4_max[i]] 
                     for i in range(nn) if res[i] == 1]
            output[0].extend(temp1)
            output[1].extend(temp2)
            output[2].extend(temp3)
            ts3 = time.time()
            print(f'Batch {i_iter}/{n_iter} for iteration {ii} finished')
        count = sum(counts)
        print(f'Iteration {ii}: ')
        print(f'\t Number of smallest intervals (one dimension): {n_interval}')
        print('results:')
        print(f'\t feasible regions:      {counts[2]/count*100} %')
        print(f'\t infeasible regions:    {counts[0]/count*100} %')
        print(f'\t indeterminate regions: {counts[1]/count*100} %')
        print(f'number of evaluations after iteration {ii}: {count_reg}')
                                                

    end = time.time()
    print(f'Total runtime: {end-start}s')
    
    print('Generating output csv file...')

    # Write all partitioned regions in the design space with the class labels 
    # into a csv file
    with open(f"test_4D_{nmax}_feasible.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x1min", "x1max", "x2min", "x2max", "x3min", "x3max", 
                         "x4min", "x4max", "label"])
        for i in range(3):
            if i == 2:
                for j in range(len(output[i])):
                    writer.writerow(output[i][j]+[i-1])
 

    with open(f"test_4D_{nmax}.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x1min", "x1max", "x2min", "x2max", "x3min", "x3max", 
                         "x4min", "x4max", "label"])
        for i in range(3):
            for j in range(len(output[i])):
                writer.writerow(output[i][j]+[i-1])
  