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

# Python code file “PITSA2G_Server.py” sets up the PITSA method for the 
# two-dimensional problem on the GPU-based server.

from numba import cuda
import numpy as np
import math
import time
import multiprocessing
from multiprocessing import Process
import setup_2G as setup
import csv

# Define the design space
x1_range = [-2*np.pi, 2*np.pi]
x2_range = [-2*np.pi, 2*np.pi]


@cuda.jit
def check(x1_min, x1_max, x2_min, x2_max, 
          result1, result2, result3, n, n_ref):
    """
    
    Function "check" checks a series of undefined 2D regions, and classify each 
    of these regions as a feasible/infeasible/indeterminate region.
    This function is designed to be performed on CUDA device
    
    x1_min, x1_max, x2_min, x2_max: cuda array
        the min or max value of x1/x2 value of each each region in the series
        
    The length of these arrays is equal to the number of undefined regions to
    be checked in this function
    
    """
    limit = 0
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n:
        # Extract the corresponding min/max x1/x2 value of a region from the 
        # input arrays based on the thread index
        x11 = x1_min[idx]
        x12 = x1_max[idx]
        x21 = x2_min[idx]
        x22 = x2_max[idx]
        # x1 as outter loop, x2 as inner loop
        for i in range(n_ref**2):
            a1 = i // n_ref
            a2 = i %  n_ref
            x11_temp = x11 + a1/n_ref * (x12-x11)
            x12_temp = x11 + (a1+1)/n_ref * (x12-x11)
            x21_temp = x21 + a2/n_ref * (x22-x21)
            x22_temp = x21 + (a2+1)/n_ref * (x22-x21)
            # Use the function defined in the setup file to calculate the range 
            # of the output of the constraint function 
            ans1_temp, ans2_temp = setup.g1(x11_temp, x12_temp, 
                                           x21_temp, x22_temp)
            if i == 0:
                ans1 = ans1_temp
                ans2 = ans2_temp
            else:
                ans1 = min(ans1, ans1_temp)
                ans2 = max(ans2, ans2_temp)
        # update the min value array
        result1[idx] = ans1
        # update the max value array
        result2[idx] = ans2
        # classify this region based on lower and upper limit and update the 
        # array of classification result
        # -1: infeasible region; 1: feasible region; 0: indeterminate region
        if ans1 > limit:
            result3[idx] = -1
        elif ans2 <= limit:
            result3[idx] = 1
        else:
            result3[idx] = 0


def mesh1(x1min, x1max, x2min, x2max, n):
    """
    
    Fucntion "mesh1" partitions a 2D region into n*n small 2D regions using the 
    meshgrid function in python.

    Parameters
    ----------
    x1min/x1max/x2min/x2max: float
        The max or min x1/x2 values of this 2D region
    n : integer
        The number of partitioned intervals in each dimension.

    Returns
    -------
    x1_min/x1_max/x2_min/x2_max: 1D array
        Array of max or min x1/x2 values of all partitioned 2D regions.
        Length of these arrays is equal to the number of new regions obtained 
        from partitioning. 


    """
    x1 = np.linspace(x1min, x1max, n+1)
    x2 = np.linspace(x2min, x2max, n+1)
    x_min = np.meshgrid(x1[:-1], x2[:-1])
    x_max = np.meshgrid(x1[1:], x2[1:])
    # flatten all meshed points and return as function outputs
    x1_min = x_min[0].flatten()
    x1_max = x_max[0].flatten()
    x2_min = x_min[1].flatten()
    x2_max = x_max[1].flatten()
    return x1_min, x1_max, x2_min, x2_max

def mesh2(list0, q, n):
    """
    This function performs mesh1 function repetitively for a series of 2D 
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
    pts = [[], [], [], []]
    for i in range(len(list0)):
        ans = mesh1(list0[i][0], list0[i][1], list0[i][2], list0[i][3], n)
        pts[0] += list(ans[0])
        pts[1] += list(ans[1])
        pts[2] += list(ans[2])
        pts[3] += list(ans[3])
    q.put(pts)




if __name__ == "__main__":
    # Set a required stopping criteria by inputting a float number
    # dx is the objective precision level of the partitioning process. 
    # The algorithm stops when the sizes of the smallest intervals on all 
    # dimensions reach a value below dx.
    dx = input('The objective level of resolution for the search process = ')
    dx = float(dx)
    
    # n_ref is the number of subintervals in each dimension using splitting
    n_ref = input('Number of refinement = ')
    n_ref = int(n_ref)

    # Number of CUDA cores available on current device 
    n_cores = 10752

    # Number of threads in each block
    tpb = 512

    # Threshold value for maximum number of undefined regions that can be 
    # analyzed in one iteration
    n_breakout = 1500000

    # number of CPU parallel threads used in the data modification process
    n_proc = 50

    # Record starting timestamp
    start = time.time()

    # Number of smallest intervals in each dimension updated in each iteration
    n_interval = 1

    # n_threshold is the threshold value used to determine whether CPU parallel
    # processing needs to be applied in data modification process
    n_threshold = 5000

    # The output list consists of three independent lists 
    # Each of them contains of min and max values of x1 and x2 of all the 
    # feasible/indeterminate/infeasible regions respectively.
    # The list is updated after each iteration, and starts with setting the 
    # entire design space as a indeterminate region
    output = [[], [[-2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi]], []]

    # Counters for infeasible/indeterminate/feasible regions respectively
    counts = [0, 0, 0]
    count_id = 0
    # The number of iterations in this algorithm (updated in each iteration)
    ii = 0
    flag = 0
    flags = [0,0,0]
    while ((x1_range[1]-x1_range[0])/n_interval>dx or 
           (x2_range[1]-x2_range[0])/n_interval>dx):
        if (x1_range[1]-x1_range[0])/n_interval<0.001 and flags[0]==0:
            flags[0] = 1
            print('Level of precision reaches 0.001')
        if (x1_range[1]-x1_range[0])/n_interval<0.0001 and flags[1]==0:
            flags[1] = 1
            print('Level of precision reaches 0.0001')
        if (x1_range[1]-x1_range[0])/n_interval<0.00001 and flags[2]==0:
            flags[2] = 1
            print('Level of precision reaches 0.00001')

        # list0 is a list of all the indeterminate regions obtained from last 
        #iteration, and these regions need to be partitioned and analyzed in 
        # the current iteration
        list0 = output[1]

        # Use the number of parallel cores to set the partition strategy in the
        # current iteration
        # n0 is the number of small intervals that one region is partitioned 
        # into in each dimension in the current iteration
        n0 = math.floor((n_cores / len(list0))**0.5)
        n0 = max(n0, 2)
        # Jump out of the current loop and go to the next loop for analysis in 
        # batches if there are too many regions to be analyzed
        if len(list0) * n0**2 > n_breakout:
            flag = 1
            break
        ts0 = time.time()

        # Update the iteration number
        ii += 1

        # Update the number of smallest itervals in each dimension
        n_interval *= n0

        # Clear the output list containing indeterminate regions
        output[1] = []

        # Update the region counters
        counts[0] *= n0**2
        counts[1] = 0
        counts[2] *= n0**2

        # The number of partitioned regions to be analyzed in this iteration.
        nn = len(list0) * n0**2
        count_id += nn

        # This part modifies all indeterminate regions into the form that can 
        # be processed with the "check" function defined above.
        # If the number of undefined regions exceed a certain threshold, apply 
        # CPU parallel processing in ths process.
        if nn >= n_threshold:
            # partition the original list of indeterminate regions based on the
            # number of threads
            ind = np.linspace(0, len(list0), n_proc+1).astype(int)
            list0_div = [list0[ind[i]:ind[i+1]] for i in range(n_proc)]
            # create parallel processes and assign regions to each process
            q = multiprocessing.Queue()
            jobs = []
            for i in range(n_proc):
                p = Process(target=mesh2, args=(list0_div[i], q, n0))
                jobs.append(p)
            for p in jobs:
                p.start()
            for p in jobs:
                p.join(timeout=1e-8)
            # extract and modify the output results
            results = [q.get() for j in jobs]
            x1_min = []
            x1_max = []
            x2_min = []
            x2_max = []
            for res in results:
                x1_min += res[0]
                x1_max += res[1]
                x2_min += res[2]
                x2_max += res[3]
        # serial processing
        else:
            x1_temp = [np.linspace(var[0], var[1], n0+1) for var in list0]
            x2_temp = [np.linspace(var[2], var[3], n0+1) for var in list0]
            xx_min = [np.meshgrid(x1_temp[i][:-1], x2_temp[i][:-1]) 
                      for i in range(len(list0))]
            xx_max = [np.meshgrid(x1_temp[i][1:], x2_temp[i][1:]) 
                      for i in range(len(list0))]
            x1_min = np.hstack([xx_min[i][0].flatten() 
                                for i in range(len(list0))])
            x1_max = np.hstack([xx_max[i][0].flatten() 
                                for i in range(len(list0))])
            x2_min = np.hstack([xx_min[i][1].flatten() 
                                for i in range(len(list0))])
            x2_max = np.hstack([xx_max[i][1].flatten() 
                                for i in range(len(list0))])
        # Define multiple empty arrays as result arrays for label collection
        res1 = cuda.device_array(nn)
        res2 = cuda.device_array(nn)
        res3 = cuda.device_array(nn)
        # Send all undefined region data to GPU
        x1_min_gpu = cuda.to_device(np.array(x1_min))
        x1_max_gpu = cuda.to_device(np.array(x1_max))
        x2_min_gpu = cuda.to_device(np.array(x2_min))
        x2_max_gpu = cuda.to_device(np.array(x2_max))
        ts1 = time.time()
        # number of blocks in the grid
        bpg = math.ceil(nn/tpb)
        # Call the "check" function to process all the undefined regions
        check[bpg, tpb](x1_min_gpu, x1_max_gpu, 
                                                  x2_min_gpu, x2_max_gpu, 
                                                  res1, res2, res3, nn, n_ref)
        # Synchronize all the threads
        cuda.synchronize()
        ts2 = time.time()
        # Copy the list containing all labels from GPU back to CPU
        res3 = res3.copy_to_host()
        # Classify all labels and obtain the number of labels in each class
        # -1: infeasible region; 0: indeterminate region; 1: feasible region
        values, counts_label = np.unique(res3, return_counts=True)
        if -1 in values:
            ind = np.where(values==-1)
            counts[0] += counts_label[ind][0]
        if 0 in values:
            ind = np.where(values==0)
            counts[1] += counts_label[ind][0]
        if 1 in values:
            ind = np.where(values==1)
            counts[2] += counts_label[ind][0]
        count = sum(counts)
        # Update the output lists with corresponding partitioned regions
        temp1 = [[x1_min[i], x1_max[i], x2_min[i], x2_max[i]] 
                 for i in range(nn) if res3[i] == -1]
        temp2 = [[x1_min[i], x1_max[i], x2_min[i], x2_max[i]] 
                 for i in range(nn) if res3[i] == 0]
        temp3 = [[x1_min[i], x1_max[i], x2_min[i], x2_max[i]] 
                 for i in range(nn) if res3[i] == 1]
        output[0].extend(temp1)
        output[1].extend(temp2)
        output[2].extend(temp3)
        
        ts3 = time.time()
        # Print basic information after each iteration
        print(f'iteration {ii}: ')
        print(f'\t Time spent so far: {time.time()-start}s')
        print(f'\t Number of smallest intervals (one dimension): {n_interval}')
        print('results:')
        print(f'\t {counts[2]/count} feasible regions')
        print(f'\t {counts[0]/count} infeasible regions')
        print(f'\t {counts[1]/count} indeterminate regions')


    # When there are too many undefnined regions to be analyzed in one 
    # iteration, the algorithm goes to this part, where all the regions are 
    # segmented into multiple batches and analyzed in batches.
    # Notably, in this part, the list that stores all indeterminate regions 
    # from the last iteration is named as "list0_tot", and a new "list0" is
    # defined in each batch containing all indeterminate regions that need 
    # to be partitioned and analyzed in the current batch.
    while flag == 1:
        if ((x1_range[1]-x1_range[0])/n_interval<=dx or 
            (x2_range[1]-x2_range[0])/n_interval<=dx):
            # 
            print((x1_range[1]-x1_range[0])/n_interval)
            break
        list0_tot = output[1]
        n0 = math.floor((n_cores / len(list0_tot))**0.5)
        n0 = max(n0, 2)
        print(n0)
        ii += 1
        n_interval *= n0
        output[1] = []
        counts[0] *= n0**2
        counts[2] *= n0**2
        counts[1] = 0
        n_iter = math.ceil(len(list0_tot) / n_breakout)
        list_ind = np.linspace(0, len(list0_tot), n_iter+1).astype(int)
        

        for i_iter in range(n_iter):
            ts0 = time.time()
            list0 = list0_tot[list_ind[i_iter]:list_ind[i_iter+1]]
            nn = len(list0) * n0 **2
            count_id += nn
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
            for res in results:
                x1_min += res[0]
                x1_max += res[1]
                x2_min += res[2]
                x2_max += res[3]
            res1 = cuda.device_array(nn)
            res2 = cuda.device_array(nn)
            res3 = cuda.device_array(nn)
            x1_min_gpu = cuda.to_device(np.array(x1_min))
            x1_max_gpu = cuda.to_device(np.array(x1_max))
            x2_min_gpu = cuda.to_device(np.array(x2_min))
            x2_max_gpu = cuda.to_device(np.array(x2_max))
            ts1 = time.time()
            # number of blocks in the grid
            bpg = math.ceil(len(x1_min)/tpb)
            check[bpg, tpb](x1_min_gpu, x1_max_gpu, x2_min_gpu, x2_max_gpu, 
                            res1, res2, res3, nn, n_ref)
            cuda.synchronize()
            ts2 = time.time()
            res3 = res3.copy_to_host()
            values, counts_label = np.unique(res3, return_counts=True)
            if -1 in values:
                ind = np.where(values==-1)
                counts[0] += counts_label[ind][0]
            if 0 in values:
                ind = np.where(values==0)
                counts[1] += counts_label[ind][0]
            if 1 in values:
                ind = np.where(values==1)
                counts[2] += counts_label[ind][0]
            temp1 = [[x1_min[i], x1_max[i], x2_min[i], x2_max[i]] 
                     for i in range(nn) if res3[i] == -1]
            temp2 = [[x1_min[i], x1_max[i], x2_min[i], x2_max[i]] 
                     for i in range(nn) if res3[i] == 0]
            temp3 = [[x1_min[i], x1_max[i], x2_min[i], x2_max[i]] 
                     for i in range(nn) if res3[i] == 1]
            output[0].extend(temp1)
            output[1].extend(temp2)
            output[2].extend(temp3)
            ts3 = time.time()
            print(f'Batch {i_iter}/{n_iter} for iteration {ii} finished')
        count = sum(counts)
        print(f'iteration {ii}: ')
        print(f'\t Time spent so far: {time.time()-start}s')
        print(f'\t Number of smallest intervals (one dimension): {n_interval}')
        print('results:')
        print(f'\t {counts[2]/count} feasible regions')
        print(f'\t {counts[0]/count} infeasible regions')
        print(f'\t {counts[1]/count} indeterminate regions')
        print(f'{count_id}')

    end = time.time()
    print(end-start)
    
    # At this step, all partitioned regions in the design space are output and 
    # stored in a csv file along with the classified labels
    with open(f"test_2D_{dx}.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x1min", "x1max", "x2min", "x2max", "label"])
        for i in range(3):
            for j in range(len(output[i])):
                writer.writerow(output[i][j]+[i-1])
