main: main.o global.o global.hpp Makefile
	nvcc -arch=sm_35 -o main main.o global.o -g

main.o: main.cpp global.hpp Makefile
	nvcc -arch=sm_35 -dc -x cu -c -o main.o main.cpp -g -O0

global.o: global.cpp global.hpp Makefile
	nvcc -arch=sm_35 -dc -x cu -c -o global.o global.cpp -g -O0
