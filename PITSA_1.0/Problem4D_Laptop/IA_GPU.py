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

# Python code file “IA_GPU.py” realizes the basic operational rules for 
# interval arithmetic on GPU-based devices.

import math
from numba import cuda, jit

# Interval arithmetic operation - addition between intervals
@cuda.jit
def add(x1min, x1max, x2min, x2max):
    return x1min+x2min, x1max+x2max

# Interval arithmetic operation - division between intervals
@cuda.jit
def minus(x1min, x1max, x2min, x2max):
    return x1min-x2max, x1max-x2min

# Interval arithmetic operation - multiplication between intervals
@cuda.jit
def multiply(x1min, x1max, x2min, x2max):
    return min(x1min*x2min, x1min*x2max, x1max*x2min, x1max*x2max), \
        max(x1min*x2min, x1min*x2max, x1max*x2min, x1max*x2max)

# Interval arithmetic operation - division between intervals
@cuda.jit
def divide(x1min, x1max, x2min, x2max):
    if x2min<0 and x2max>0:
        return -math.inf, math.inf
    elif x2min == 0:
        if x1min>=0:
            return x1min/x2max, math.inf
        elif x1max<=0:
            return -math.inf, x1max/x2max
        else:
            return -math.inf, math.inf
    elif x2max==0:
        if x1min>=0:
            return -math.inf, x1min/x2min
        elif x1max<=0:
            return x1max/x2min, math.inf
        else:
            return -math.inf, math.inf
    else:
        temp1, temp2 = multiply(x1min, x1max, 1/x2max, 1/x2min)
        return temp1, temp2

# Interval arithmetic operation - power
# n can only be positive integers
@cuda.jit
def power(xmin, xmax, n):
    if xmin<0 and xmax>0 and n%2==0:
        return 0, max(xmin**n, xmax**n)
    else:
        return min(xmin**n, xmax**n), max(xmin**n, xmax**n)

# Interval arithmetic operation- multiplication between an interval and a real 
# number
@cuda.jit
def times(xmin, xmax, n):
    return min(xmin*n, xmax*n), max(xmin*n, xmax*n)

# Interval arithmetic operation - addition between an intevral and a real 
# number
@cuda.jit
def plus(xmin, xmax, n):
    return xmin+n, xmax+n

# Interval arithmetic operation - sine function
@cuda.jit
def sine(xmin, xmax):
    ans1 = min(math.sin(xmin), math.sin(xmax))
    ans2 = max(math.sin(xmin), math.sin(xmax))
    temp1 = xmin//(math.pi/2)
    temp2 = xmax//(math.pi/2)
    if (temp1+1)//4 != (temp2+1)//4:
        ans1 = -1
    if (temp1-1)//4 != (temp2-1)//4:
        ans2 = 1
    return ans1, ans2

# Interval arithmetic operation - cosine function
@cuda.jit
def cosine(xmin, xmax):
    ans1 = min(math.cos(xmin), math.cos(xmax))
    ans2 = max(math.cos(xmin), math.cos(xmax))
    temp1 = xmin//(math.pi/2)
    temp2 = xmax//(math.pi/2)
    if (temp1+2)//4 != (temp2+2)//4:
        ans1 = -1
    if (temp1)//4 != (temp2)//4:
        ans2 = 1
    return ans1, ans2

# Interval arithmetic operation - exponential
@cuda.jit
def exp(xmin, xmax):
    return math.exp(xmin), math.exp(xmax)