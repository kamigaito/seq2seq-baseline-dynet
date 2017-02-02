TARGET_TRAIN="./bin/s2s-train"
TARGET_PRED="./bin/s2s-pred"
CC=g++
OPT=-pthread -std=c++11 -lboost_program_options -lboost_filesystem -lboost_system -lboost_iostreams -g -O3
INCLUDES=-I./include -I${BOOST_ROOT}/include -I${PATH_TO_EIGEN} -I${PATH_TO_CNN} -I${PATH_TO_CUDA}/include
LIBS=-L./lib -L${BOOST_ROOT}/lib -L${PATH_TO_CUDA}/cnn
SRC=$(shell ls lib/*.cpp)
HED=$(shell ls include/*.h)
TRG=$(shell ls bin/*.cpp)
OBJ_TRAIN=$(SRC:train.cpp=train.o)
OBJ_PRED=$(SRC:predict.cpp=predict.o)

all:$(TARGET_TRAIN) $(TARGET_PRED)
$(TARGET_TRAIN):$(OBJ)
	$(CC) $(OPT) $(LIBS) $(INCLUDES) -o $(TARGET_TRAIN) $(OBJ)
$(TARGET_PRED):$(OBJ)
	$(CC) $(OPT) $(LIBS) $(INCLUDES) -o $(TARGET_TRAIN) $(OBJ)

.cpp.o:
	$(CC) $(OPT) $(LIBS) $(INCLUDES) -c $< -o $@

makefile.depend:
	g++ -MM -MG $(SRC) $(OPT) > makefile.depend

clean:
	rm ./lib/*.o ./makefile.depend

-include makefile.depend
