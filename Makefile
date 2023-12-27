CXX = clang++
CXXFLAGS += -Iinclude
CXXFLAGS += -stdlib=libstdc++ -std=c++17

SRC_DIR = src
BUILD_DIR = build
OBJS_DIR = $(BUILD_DIR)/objs

SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(patsubst $(SRC_DIR)/%.cpp, $(OBJS_DIR)/%.o, $(SRC))

EXE = $(BUILD_DIR)/edge

all: dirs $(OBJS) edge

run:
	$(EXE)

crun: edge
	$(EXE)

edge: $(OBJ)
	$(CXX) $(OBJ) $(CXXFLAGS) -o $(EXE)

$(OBJS_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

dirs:
	mkdir -p ./$(BUILD_DIR) ./$(OBJS_DIR)
