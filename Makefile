CXX = g++
CXXFLAGS = -Wall -Weffc++ -std=c++11 -c -Iinclude

OBJS = bin/Main.o bin/Model.o bin/Server.o bin/GPU.o bin/MemoryBlock.o

all: bin/server

bin/server: $(OBJS)
	$(CXX) -o bin/server $(OBJS)

bin/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY: all clean
clean:
	rm -f bin/*