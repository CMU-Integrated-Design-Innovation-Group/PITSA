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

# Python code file “setup_2G.py” defines the constraint function in the 
# two-dimensional problem in the form of interval arithmetic.

from numba import cuda, jit
import IA_GPU as ia
import numpy as np
import math

# The two following functions can round up or round down a given float number 
# to a given decimal level
@cuda.jit
def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

@cuda.jit
def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


# Outward-round an interval to a certain level of decimal/significant numbers
@cuda.jit
def outward_rounding(num1, num2, n, flag):
    # num1: lower limit of the interval
    # num2: upper limit of the interval
    # n: number of decimal/significant level
    # flag: flag = 1 for significant-number-based outward rounding
    #       flag = 2 for decimal-number-based outward rounding
    if num1 == 0:
        ans1 = 0
    elif num1 < 0:
        if flag == 1:
            ans1 = -round_up(-num1, n-1-math.floor(math.log10(-num1)))
        if flag == 2:
            ans1 = -round_up(-num1, n)
    else:
        if flag == 1:
            ans1 = round_down(num1, n-1-math.floor(math.log10(num1)))
        if flag == 2:
            ans1 = round_down(num1, n)

    if num2 == 0:
        ans2 = 0
    elif num2 < 0:
        if flag == 1:
            ans2 = -round_down(-num2, n-1-math.floor(math.log10(-num2)))
        if flag == 2:
            ans2 = -round_down(-num2, n)
    else:
        if flag == 1:
            ans2 = round_up(num2, n-1-math.floor(math.log10(num2)))
        if flag == 2:
            ans2 = round_up(num2, n)
    
    return ans1, ans2

@cuda.jit
# g1(x1, x2) = (x1-x2)^2 + sin(x1)*e^[1-cos(x2)]^2 + cos(x2)*e^[1-sin(x1)]^2
# Function "g1" defines the cosntraint function in the two-dimensional problem
# using interval arithmetic, and each step is corresponding to the code list of
# this function
def g1(x1min, x1max, x2min, x2max):
    n1 = 8
    n2 = 8
    flag = 1
    # G1_1 = x1 - x2
    G1_1l, G1_1u = ia.minus(x1min, x1max, x2min, x2max)
    G1_1l, G1_1l = outward_rounding(G1_1l, G1_1u, n1, flag)
    # G1_2 = (G1_1)^2
    G1_2l, G1_2u = ia.power(G1_1l, G1_1u, 2)
    G1_2l, G1_2u = outward_rounding(G1_2l, G1_2u, n1, flag)
    # G1_3 = cos(x2)
    G1_3l, G1_3u = ia.cosine(x2min, x2max)
    G1_3l, G1_3u = outward_rounding(G1_3l, G1_3u, n1, flag)
    # G1_4 = 1 - G1_3
    G1_4l, G1_4u = ia.minus(1, 1, G1_3l, G1_3u)
    G1_4l, G1_4u = outward_rounding(G1_4l, G1_4u, n1, flag)
    # G1_5 = G1_4^2
    G1_5l, G1_5u = ia.power(G1_4l, G1_4u, 2)
    G1_5l, G1_5u = outward_rounding(G1_5l, G1_5u, n1, flag)
    # G1_6 = e^G1_5
    G1_6l, G1_6u = ia.exp(G1_5l, G1_5u)
    G1_6l, G1_6u = outward_rounding(G1_6l, G1_6u, n1, flag)
    # G1_7 = sin(x1)
    G1_7l, G1_7u = ia.sine(x1min, x1max)
    G1_7l, G1_7u = outward_rounding(G1_7l, G1_7u, n1, flag)
    # G1_8 = G1_6 * G1_7
    G1_8l, G1_8u = ia.multiply(G1_7l, G1_7u, G1_6l, G1_6u)
    G1_8l, G1_8u = outward_rounding(G1_8l, G1_8u, n1, flag)
    # G1_9 = 1 - G1_7
    G1_9l, G1_9u = ia.minus(1, 1, G1_7l, G1_7u)
    G1_9l, G1_9u = outward_rounding(G1_9l, G1_9u, n1, flag)
    # G1_10 = G1_9^2
    G1_10l, G1_10u = ia.power(G1_9l, G1_9u, 2)
    G1_10l, G1_10u = outward_rounding(G1_10l, G1_10u, n1, flag)
    # G1_11 = e^G1_10
    G1_11l, G1_11u = ia.exp(G1_10l, G1_10u)
    G1_11l, G1_11u = outward_rounding(G1_11l, G1_11u, n1, flag)
    # G1_12 = G1_3 * G1_11
    G1_12l, G1_12u = ia.multiply(G1_3l, G1_3u, G1_11l, G1_11u)
    G1_12l, G1_12u = outward_rounding(G1_12l, G1_12u, n1, flag)
    # G1_13 = G1_2 + G1_8
    G1_13l, G1_13u = ia.add(G1_2l, G1_2u, G1_8l, G1_8u)
    G1_13l, G1_13u = outward_rounding(G1_13l, G1_13u, n1, flag)
    # g1 = G1_12 + G1_13
    g1l, g1u = ia.add(G1_12l, G1_12u, G1_13l, G1_13u)
    g1l, g1u = outward_rounding(g1l, g1u, n1, flag)
    return g1l, g1u