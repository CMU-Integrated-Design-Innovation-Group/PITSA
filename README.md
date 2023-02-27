# PITSA

The PITSA project provides the code lists and the Python source code to implement **a GPU-based iterative parallel classification method** for two continuous constraint satisfaction problems (i.e., two case studies). The results of these two problems are also included in the project. 

The PITSA project is authored by Wangchuan Feng, [Guanglu Zhang](https://www.researchgate.net/profile/Guanglu-Zhang), and [Jonathan Cagan](https://www.meche.engineering.cmu.edu/directory/bios/cagan-jonathan.html), from [the Integrated Design Innovation Group (IDIG)](https://www.cmu.edu/me/idig/) at Carnegie Mellon University.

## Contents
1. [Directory Tree](https://github.com/CMU-Integrated-Design-Innovation-Group/PITSA#directory-tree)
2. [Folder and File Descriptions](https://github.com/CMU-Integrated-Design-Innovation-Group/PITSA#folder-and-file-descriptions)
3. [Environment](https://github.com/CMU-Integrated-Design-Innovation-Group/PITSA#environment)
4. [Instructions](https://github.com/CMU-Integrated-Design-Innovation-Group/PITSA#instructions)
5. [Citation](https://github.com/CMU-Integrated-Design-Innovation-Group/PITSA#citation)
6. [License](https://github.com/CMU-Integrated-Design-Innovation-Group/PITSA#license)

## Directory Tree
![3](https://user-images.githubusercontent.com/117691203/221443444-4d787ee2-bb3d-474e-b1fe-e77186f74de4.png)

## Folder and File Descriptions

### RBS_codes 
The folder “RBS_codes” constains the Python scripts and the results for the two  continuous constraint satisfaction problems. These Python scripts are executed using two computing resources, including a laptop with one GPU and a server with one GPU. 

#### Problem2D_Laptop
The folder “Problem2D_Laptop” contains the following five Python scripts for implementing the sample-based exhaustive search method, the region-based exhaustive search method, and the GPU-based iterative parallel classification method to solve the two-dimensional continuous constraint satisfaction problem involving birds function. These Python scripts are executed using a laptop with one GPU.

-  IA_GPU.py: The Python script “IA_GPU.py” realizes several basic interval arithmetic operations on GPU.

-  setup_2G.py: The Python script “setup_2G.py” defines the constraint function of the two-dimensional problem based on interval arithmetic.

-  SBES2G_Laptop.py: The Python script “SBES2G_Laptop.py” sets up the sample-based exhaustive search method to solve the two-dimensional problem using the laptop with one GPU.

-  RBES2G_Laptop.py: The Python script “RBES2G_Laptop.py” sets up the region-based exhaustive search method to solve the two-dimensional problem using the laptop with one GPU.

-  PITSA2G_Laptop.py: The Python script “PITSA2G_Laptop.py” sets up the GPU-based iterative parallel classification method to solve the two-dimensional problem using the laptop with one GPU. The Python script “PITSA2G_Laptop.py” calls “IA_GPU.py” and “setup_2G.py” when it is executed. 

#### Problem2D_Server
The folder “Problem2D_Server” contains the following five Python scripts for implementing the sample-based exhaustive search method, the region-based exhaustive search method, and the GPU-based iterative parallel classification method to solve the two-dimensional continuous constraint satisfaction problem involving birds function. These Python scripts are executed using a server with one GPU. A .csv data file is also included as the result derived from the GPU-based iterative parallel classification method.

-  IA_GPU.py: The Python script “IA_GPU.py” realizes several basic interval arithmetic operations on GPU.

-  setup_2G.py: The Python script “setup_2G.py” defines the constraint function of the two-dimensional problem based on interval arithmetic.

-  SBES2G_Server.py: The Python script “SBES2G_Server.py” sets up the sample-based exhaustive search method to solve the two-dimensional problem using the server with one GPU.

-  RBES2G_Server.py: The Python script “RBES2G_Server.py” sets up the region-based exhaustive search method to solve the two-dimensional problem using the server with one GPU.

-  PITSA2G_Server.py: The Python script “PITSA2G_Server.py” sets up the GPU-based iterative parallel classification method to solve the two-dimensional problem using the server with one GPU. The Python script “PITSA2G_Server.py” calls “IA_GPU.py” and “setup_2G.py” when it is executed.

-  test_2D_196608.csv: The csv file “test_2D_196608.csv” includes the sets of intervals as the result derived from the the GPU-based iterative parallel classification method. Each set of two intervals corresponds to one classified region in the design space. 

#### Problem4D_Laptop
The folder “Problem4D_Laptop” contains the following six Python scripts for implementing the sample-based exhaustive search method, the region-based exhaustive search method, and the GPU-based iterative parallel classification method to solve the four-dimensional continuous constraint satisfaction problem of welded beam design. These Python scripts are executed using a laptop with one GPU.



#### Problem4D_Server
The folder “Problem4D_Laptop” contains the following six Python scripts for implementing the sample-based exhaustive search method, the region-based exhaustive search method, and the GPU-based iterative parallel classification method to solve the four-dimensional continuous constraint satisfaction problem of welded beam design. These Python scripts are executed using a server with one GPU.

## Environment

## Instructions

## Citation

## License
PITSA is freely available for academic or non-profit organization noncommercial research only. Please check [the license file](https://github.com/CMU-Integrated-Design-Innovation-Group/PITSA/blob/main/LICENSE) for further details. If you are interested in a commercial license, please contact [CMU Center for Technology Transfer and Enterprise Creation](https://www.cmu.edu/cttec/contact-us/index.html) at **innovation@cmu.edu**.


