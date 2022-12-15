test: test.cpp
	nvcc -x cu -o test test.cpp -g -O0 --extended-lambda -std=c++17  --keep -G

lambda: lambda.cpp Makefile
	clang++ -xcuda lambda.cpp -o lambda --std=c++17 --cuda-gpu-arch=sm_70 -L/usr/local/cuda-11.2/targets/x86_64-linux/lib/ -lcudart_static -ldl -lrt -pthread -Wno-unknown-cuda-version


main: main.o global.o global.hpp Makefile
	nvcc -arch=sm_60 -o main main.o global.o -g

main.o: main.cpp global.hpp Makefile
	nvcc --extended-lambda -dc -x cu -c -o main.o main.cpp -g -O0

global.o: global.cpp global.hpp Makefile
	nvcc -arch=sm_60 -dc -x cu -c -o global.o global.cpp -g -O0
