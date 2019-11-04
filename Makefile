main: main.o global.o global.hpp Makefile
	nvcc -o main main.o global.o -g -O0

main.o: main.cpp Makefile
	nvcc -x cu -c -o main.o main.cpp -g -O0

global.o: global.cpp global.hpp Makefile
	nvcc -x cu -c -o global.o global.cpp -g -O0
