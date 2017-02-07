TARGET="./bin/seq2seq"
CC=g++
OPT=-pthread -std=c++11 -lboost_program_options -lboost_filesystem -lboost_system -lboost_iostreams -lboost_serialization -lgdynet -ldynetcuda -g -O3
INCLUDES=-I./include -I${BOOST_ROOT}/include -I${PATH_TO_EIGEN} -I/home/lr/kamigaito/Apps/dynet -I${PATH_TO_CUDA}/include
LIBS=-L./lib -L${BOOST_ROOT}/lib -L${PATH_TO_CUDA}/lib -L/home/lr/kamigaito/Apps/dynet/build-cuda/dynet
SRC=$(shell ls lib/*.cpp)
HED=$(shell ls include/*.h)
OBJ=$(SRC:seq2seq.cpp=seq2seq.o)

all:$(TARGET)
$(TARGET):$(OBJ)
	$(CC) $(OPT) $(LIBS) $(INCLUDES) -o $(TARGET) $(OBJ)

.cpp.o:
	$(CC) $(OPT) $(LIBS) $(INCLUDES) -c $< -o $@

makefile.depend:
	g++ -MM -MG $(SRC) $(OPT) > makefile.depend

clean:
	rm ./lib/*.o ./makefile.depend

-include makefile.depend
