CXX = icc
# CXX2 = gcc
CXXFLAGS2 = -O3 -Wall
#CXXFLAGS3 = -arch=sm_20 -use_fast_math -O3
CXXFLAGS3 = -O3 
#CXXFLAGS3 =
TARGET1= dotprod_openmp

all : $(TARGET1)
    
$(TARGET1) : dotprod_openmp.c kernel.c solution.h
	$(CXX) $(CXXFLAGS3) -lm -o $(TARGET1) dotprod_openmp.c kernel.c -lm -fopenmp
clean : 
	rm -f $(TARGET1)
