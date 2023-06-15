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

# Python code file “setup_4G_SBES.py” defines all constraint functions in the 
# four-dimensional problem in the form of floating number arithmetic.

import numpy as np
import time
from numba import cuda, jit

Pc = 6000
Lc = 14
Ec = 30*10**6
Gc = 12*10**6
Sc1 = 0.10471
Sc2 = 0.04811

@cuda.jit
def g1(x1, x2, x3, x4):
    J = 2*(2**0.5*x1*x2*(x2**2/12+((x1+x3)/2)**2))
    R = (x2**2/4 + ((x1+x3)/2)**2)**0.5
    t1 = Pc/(2**0.5*x1*x2)
    M = Pc*(Lc+x2/2)
    t2 = M*R/J
    t = (t1**2 + 2*t1*t2*x2/(2*R) +t2**2)**0.5
    return t-13600

@cuda.jit
def g2(x1, x2, x3, x4):
    return 6*Pc*Lc/(x4*x3**2)-30000

@cuda.jit
def g3(x1, x2, x3, x4):
    return x1-x4

@cuda.jit
def g4(x1, x2, x3, x4):
    return Sc1*x1**2*x2 + Sc2*x3*x4*(Lc+x2)-5

@cuda.jit
def g5(x1, x2, x3, x4):
    return 0.125-x1

@cuda.jit
def g6(x1, x2, x3, x4):
    return 4*Pc*Lc**3/(Ec*x3**3*x4)-0.25

@cuda.jit
def g7(x1, x2, x3, x4):
    P = 4.013*(Ec*Gc*x3**2*x4**6/36)**0.5/Lc**2*(1-0.5*x3/Lc*(0.25*Ec/Gc)**0.5)
    return Pc-P

@cuda.jit
def g8(x1, x2, x3, x4):
    return (1+Sc1)*x1**2*x2+Sc2*x3*x4*(14+x2)-3.5


