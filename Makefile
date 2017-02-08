TARGET="./bin/seq2seq"
CC=nvcc
LINK=-lboost_program_options -lboost_filesystem -lboost_system -lboost_regex -lboost_iostreams -lboost_serialization -lgdynet -ldynetcuda -lcublas -lcudart
OPT=-std=c++11 -g -O3
INCLUDES=-I./include -I${BOOST_ROOT}/include -I${PATH_TO_EIGEN} -I${PATH_TO_DYNET} -I${PATH_TO_CUDA}/include
LIBS=-L./lib -L${BOOST_ROOT}/lib -L${BOOST_ROOT}/libs -L${PATH_TO_EIGEN}/build/lib -L${PATH_TO_DYNET}/build-cuda/dynet -L${PATH_TO_CUDA}/lib64 -L${PATH_TO_CUDA}/lib
SRC=$(shell ls lib/*.cpp)
HED=$(shell ls include/*.hpp)
OBJ=$(SRC:seq2seq.cpp=seq2seq.o)

all:$(TARGET)
$(TARGET):$(OBJ)
	$(CC) $(OPT) $(LIBS) $(INCLUDES) -o $(TARGET) $(OBJ) $(LINK)

.cpp.o:
	$(CC) $(OPT) $(LIBS) $(INCLUDES) -c $< -o $@

makefile.depend:
	g++ -MM -MG $(SRC) $(OPT) $(LINK) > makefile.depend

clean:
	rm ./lib/*.o ./makefile.depend

-include makefile.depend
