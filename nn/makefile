CXXFLAGS = -I eigen-3.4.0/Eigen
CXX = g++   

TARGET = main
SRC = main.cpp

all: $(TARGET)

$(TARGET):$(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
