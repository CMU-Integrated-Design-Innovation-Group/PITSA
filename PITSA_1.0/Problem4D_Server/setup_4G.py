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

# Python code file “setup_4G.py” defines all constraint functions in the 
# four-dimensional problem in the form of interval arithmetic.

from numba import cuda, jit
import IA_GPU as ia
import numpy as np
import math

Pc = 6000
Lc = 14
Ec = 30*10**6
Gc = 12*10**6
Sc1 = 0.10471
Sc2 = 0.04811
M = 3.5

# The two following functions round a float number up or down to the nearest 
# float number with an objective decimal level defined in the function input

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

n_or = 8
flag = 1


@cuda.jit
def g1(x1min, x1max, x2min, x2max, x3min, x3max, x4min, x4max):
    G1_1l, G1_1u = ia.multiply(x1min, x1max, x2min, x2max)
    G1_1l, G1_1u = outward_rounding(G1_1l, G1_1u, n_or, flag)
    t1l, t1u = ia.divide(Pc/2**0.5, Pc/2**0.5, G1_1l, G1_1u)
    t1l, t1u = outward_rounding(t1l, t1u, n_or, flag)
    G1_2l, G1_2u = ia.times(x2min, x2max, 0.5)
    G1_2l, G1_2u = outward_rounding(G1_2l, G1_2u, n_or, flag)
    G1_3l, G1_3u = ia.plus(G1_2l, G1_2u, Lc)
    G1_3l, G1_3u = outward_rounding(G1_3l, G1_3u, n_or, flag)
    Ml, Mu = ia.times(G1_3l, G1_3u, Pc)
    Ml, Mu = outward_rounding(Ml, Mu, n_or, flag)
    G1_4l, G1_4u = ia.power(x2min, x2max, 2)
    G1_4l, G1_4u = outward_rounding(G1_4l, G1_4u, n_or, flag)
    G1_5l, G1_5u = ia.times(G1_4l, G1_4u, 0.25)
    G1_5l, G1_5u = outward_rounding(G1_5l, G1_5u, n_or, flag)
    G1_6l, G1_6u = ia.add(x1min, x1max, x3min, x3max)
    G1_6l, G1_6u = outward_rounding(G1_6l, G1_6u, n_or, flag)
    G1_7l, G1_7u = ia.power(G1_6l, G1_6u, 2)
    G1_7l, G1_7u = outward_rounding(G1_7l, G1_7u, n_or, flag)
    G1_8l, G1_8u = ia.times(G1_7l, G1_7u, 0.25)
    G1_8l, G1_8u = outward_rounding(G1_8l, G1_8u, n_or, flag)
    G1_9l, G1_9u = ia.add(G1_5l, G1_5u, G1_8l, G1_8u)
    G1_9l, G1_9u = outward_rounding(G1_9l, G1_9u, n_or, flag)
    Rl, Ru = ia.power(G1_9l, G1_9u, 0.5)
    Rl, Ru = outward_rounding(Rl, Ru, n_or, flag)
    G1_10l, G1_10u = ia.power(x2min, x2max, 2)
    G1_10l, G1_10u = outward_rounding(G1_10l, G1_10u, n_or, flag)
    G1_11l, G1_11u = ia.times(G1_10l, G1_10u, 1/12)
    G1_11l, G1_11u = outward_rounding(G1_11l, G1_11u, n_or, flag)
    G1_12l, G1_12u = ia.multiply(x1min, x1max, x2min, x2max)
    G1_12l, G1_12u = outward_rounding(G1_12l, G1_12u, n_or, flag)
    G1_13l, G1_13u = ia.times(G1_12l, G1_12u, 2*(2**0.5))
    G1_13l, G1_13u = outward_rounding(G1_13l, G1_13u, n_or, flag)
    G1_14l, G1_14u = ia.add(G1_11l, G1_11u, G1_8l, G1_8u)
    G1_14l, G1_14u = outward_rounding(G1_14l, G1_14u, n_or, flag)
    Jl, Ju = ia.multiply(G1_14l, G1_14u, G1_13l, G1_13u)
    Jl, Ju = outward_rounding(Jl, Ju, n_or, flag)
    G1_15l, G1_15u = ia.multiply(Ml, Mu, Rl, Ru)
    G1_15l, G1_15u = outward_rounding(G1_15l, G1_15u, n_or, flag)
    t2l, t2u = ia.divide(G1_15l, G1_15u, Jl, Ju)
    t2l, t2u = outward_rounding(t2l, t2u, n_or, flag)
    G1_16l, G1_16u = ia.power(t1l, t1u, 2)
    G1_16l, G1_16u = outward_rounding(G1_16l, G1_16u, n_or, flag)
    G1_17l, G1_17u = ia.power(t2l, t2u, 2)
    G1_17l, G1_17u = outward_rounding(G1_17l, G1_17u, n_or, flag)
    G1_18l, G1_18u = ia.add(G1_16l, G1_16u, G1_17l, G1_17u)
    G1_18l, G1_18u = outward_rounding(G1_18l, G1_18u, n_or, flag)
    G1_19l, G1_19u = ia.multiply(t1l, t1u, t2l, t2u)
    G1_19l, G1_19u = outward_rounding(G1_19l, G1_19u, n_or, flag)
    G1_20l, G1_20u = ia.divide(x2min, x2max, Rl, Ru)
    G1_20l, G1_20u = outward_rounding(G1_20l, G1_20u, n_or, flag)
    G1_21l, G1_21u = ia.multiply(G1_19l, G1_19u, G1_20l, G1_20u)
    G1_21l, G1_21u = outward_rounding(G1_21l, G1_21u, n_or, flag)
    G1_22l, G1_22u = ia.add(G1_18l, G1_18u, G1_21l, G1_21u)
    G1_22l, G1_22u = outward_rounding(G1_22l, G1_22u, n_or, flag)
    tl, tu = ia.power(G1_22l, G1_22u, 0.5)
    tl, tu = outward_rounding(tl, tu, n_or, flag)
    g1l, g1u = ia.plus(tl, tu, -13600)
    g1l, g1u = outward_rounding(g1l, g1u, n_or, flag)
    return g1l, g1u


@cuda.jit
def g2(x1min, x1max, x2min, x2max, x3min, x3max, x4min, x4max):
    G2_1l, G2_1u = ia.power(x3min, x3max, 2)
    G2_1l, G2_1u = outward_rounding(G2_1l, G2_1u, n_or, flag)
    G2_2l, G2_2u = ia.multiply(x4min, x4max, G2_1l, G2_1u)
    G2_2l, G2_2u = outward_rounding(G2_2l, G2_2u, n_or, flag)
    G2_3l, G2_3u = ia.divide(6*Pc*Lc, 6*Pc*Lc, G2_2l, G2_2u)
    G2_3l, G2_3u = outward_rounding(G2_3l, G2_3u, n_or, flag)
    g2l, g2u = ia.minus(G2_3l, G2_3u, 30000, 30000)
    g2l, g2u = outward_rounding(g2l, g2u, n_or, flag)
    return g2l, g2u

@cuda.jit
def g3(x1min, x1max, x2min, x2max, x3min, x3max, x4min, x4max):
    g3l, g3u = ia.minus(x1min, x1max, x4min, x4max)
    g3l, g3u = outward_rounding(g3l, g3u, n_or, flag)
    return g3l, g3u

@cuda.jit
def g4(x1min, x1max, x2min, x2max, x3min, x3max, x4min, x4max):
    G4_1l, G4_1u = ia.power(x1min, x1max, 2)
    G4_1l, G4_1u = outward_rounding(G4_1l, G4_1u, n_or, flag)
    G4_2l, G4_2u = ia.multiply(G4_1l, G4_1u, x2min, x2max)
    G4_2l, G4_2u = outward_rounding(G4_2l, G4_2u, n_or, flag)
    G4_3l, G4_3u = ia.times(G4_2l, G4_2u, Sc1)
    G4_3l, G4_3u = outward_rounding(G4_3l, G4_3u, n_or, flag)
    G4_4l, G4_4u = ia.multiply(x3min, x3max, x4min, x4max)
    G4_4l, G4_4u = outward_rounding(G4_4l, G4_4u, n_or, flag)
    G4_5l, G4_5u = ia.times(G4_4l, G4_4u, Sc2)
    G4_5l, G4_5u = outward_rounding(G4_5l, G4_5u, n_or, flag)
    G4_6l, G4_6u = ia.plus(x2min, x2max, Lc)
    G4_6l, G4_6u = outward_rounding(G4_6l, G4_6u, n_or, flag)
    G4_7l, G4_7u = ia.multiply(G4_5l, G4_5u, G4_6l, G4_6u)
    G4_7l, G4_7u = outward_rounding(G4_7l, G4_7u, n_or, flag)
    G4_8l, G4_8u = ia.add(G4_3l, G4_3u, G4_7l, G4_7u)
    G4_8l, G4_8u = outward_rounding(G4_8l, G4_8u, n_or, flag)
    g4l, g4u = ia.plus(G4_8l, G4_8u, -5)
    g4l, g4u = outward_rounding(g4l, g4u, n_or, flag)
    return g4l, g4u

@cuda.jit
def g5(x1min, x1max, x2min, x2max, x3min, x3max, x4min, x4max):
    g5l, g5u = ia.minus(0.125, 0.125, x1min, x1max)
    g5l, g5u = outward_rounding(g5l, g5u, n_or, flag)
    return g5l, g5u

@cuda.jit
def g6(x1min, x1max, x2min, x2max, x3min, x3max, x4min, x4max):
    G6_1l, G6_1u = ia.power(x3min, x3max, 3)
    G6_1l, G6_1u = outward_rounding(G6_1l, G6_1u, n_or, flag)
    G6_2l, G6_2u = ia.multiply(G6_1l, G6_1u, x4min, x4max)
    G6_2l, G6_2u = outward_rounding(G6_2l, G6_2u, n_or, flag)
    G6_3l, G6_3u = ia.divide(1, 1, G6_2l, G6_2u)
    G6_3l, G6_3u = outward_rounding(G6_3l, G6_3u, n_or, flag)
    G6_4l, G6_4u = ia.times(G6_3l, G6_3u, 4*Pc*Lc**3/Ec)
    G6_4l, G6_4u = outward_rounding(G6_4l, G6_4u, n_or, flag)
    g6l, g6u = ia.plus(G6_4l, G6_4u, -0.25)
    g6l, g6u = outward_rounding(g6l, g6u, n_or, flag)
    return g6l, g6u

@cuda.jit
def g7(x1min, x1max, x2min, x2max, x3min, x3max, x4min, x4max):
    G7_1l, G7_1u = ia.power(x3min, x3max, 2)
    G7_1l, G7_1u = outward_rounding(G7_1l, G7_1u, n_or, flag)
    G7_2l, G7_2u = ia.power(x4min, x4max, 6)
    G7_2l, G7_2u = outward_rounding(G7_2l, G7_2u, n_or, flag)
    G7_3l, G7_3u = ia.multiply(G7_1l, G7_1u, G7_2l, G7_2u)
    G7_3l, G7_3u = outward_rounding(G7_3l, G7_3u, n_or, flag)
    G7_4l, G7_4u = ia.times(G7_3l, G7_3u, 1/36)
    G7_4l, G7_4u = outward_rounding(G7_4l, G7_4u, n_or, flag)
    G7_5l, G7_5u = ia.power(G7_4l, G7_4u, 0.5)
    G7_5l, G7_5u = outward_rounding(G7_5l, G7_5u, n_or, flag)
    G7_6l, G7_6u = ia.times(G7_5l, G7_5u, 4.013*(Ec*Gc)**0.5/Lc**2)
    G7_6l, G7_6u = outward_rounding(G7_6l, G7_6u, n_or, flag)
    G7_7l, G7_7u = ia.times(x3min, x3max, (Ec/(4*Gc))**0.5/(2*Lc))
    G7_7l, G7_7u = outward_rounding(G7_7l, G7_7u, n_or, flag)
    G7_8l, G7_8u = ia.minus(1, 1, G7_7l, G7_7u)
    G7_8l, G7_8u = outward_rounding(G7_8l, G7_8u, n_or, flag)
    G7_9l, G7_9u = ia.multiply(G7_6l, G7_6u, G7_8l, G7_8u)
    G7_9l, G7_9u = outward_rounding(G7_9l, G7_9u, n_or, flag)
    g7l, g7u = ia.minus(Pc, Pc, G7_9l, G7_9u)
    g7l, g7u = outward_rounding(g7l, g7u, n_or, flag)
    return g7l, g7u

@cuda.jit
def g8(x1min, x1max, x2min, x2max, x3min, x3max, x4min, x4max):
    G8_1l, G8_1u = ia.power(x1min, x1max, 2)
    G8_1l, G8_1u = outward_rounding(G8_1l, G8_1u, n_or, flag)
    G8_2l, G8_2u = ia.times(x2min, x2max, (1+Sc1))
    G8_2l, G8_2u = outward_rounding(G8_2l, G8_2u, n_or, flag)
    G8_3l, G8_3u = ia.multiply(G8_1l, G8_1u, G8_2l, G8_2u)
    G8_3l, G8_3u = outward_rounding(G8_3l, G8_3u, n_or, flag)
    G8_4l, G8_4u = ia.multiply(x3min, x3max, x4min, x4max)
    G8_4l, G8_4u = outward_rounding(G8_4l, G8_4u, n_or, flag)
    G8_5l, G8_5u = ia.times(G8_4l, G8_4u, Sc2)
    G8_5l, G8_5u = outward_rounding(G8_5l, G8_5u, n_or, flag)
    G8_6l, G8_6u = ia.plus(x2min, x2max, Lc)
    G8_6l, G8_6u = outward_rounding(G8_6l, G8_6u, n_or, flag)
    G8_7l, G8_7u = ia.multiply(G8_5l, G8_5u, G8_6l, G8_6u)
    G8_7l, G8_7u = outward_rounding(G8_7l, G8_7u, n_or, flag)
    G8_8l, G8_8u = ia.add(G8_3l, G8_3u, G8_7l, G8_7u)
    G8_8l, G8_8u = outward_rounding(G8_8l, G8_8u, n_or, flag)
    g8l, g8u = ia.plus(G8_8l, G8_8u, -M)
    g8l, g8u = outward_rounding(g8l, g8u, n_or, flag)
    return g8l, g8u
