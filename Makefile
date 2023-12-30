CXX = clang++
CXXFLAGS += -Iinclude -g
CXXFLAGS += -stdlib=libstdc++ -std=c++17

DBG = gdb
VAL = valgrind

SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
OBJS_DIR = $(BUILD_DIR)/objs

SRC = $(wildcard $(SRC_DIR)/*.cpp)
HEADERS = $(wildcard $(INCLUDE_DIR)/*.h) 
OBJ = $(patsubst $(SRC_DIR)/%.cpp, $(OBJS_DIR)/%.o, $(SRC))

EXE = $(BUILD_DIR)/edge
TEST_FILE = test_file.edge

all: dirs $(OBJS) edge

RUN_ARGS = $(EXE) $(TEST_FILE)
VALGRIND_ARGS = --leak-check=full

run:
	$(RUN_ARGS)

crun: edge
	$(RUN_ARGS)

edge: $(OBJ)
	$(CXX) $(OBJ) $(CXXFLAGS) -o $(EXE)

$(OBJS_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

dirs:
	mkdir -p ./$(BUILD_DIR) ./$(OBJS_DIR)

clean:
	rm $(EXE) $(OBJ)

format:
	clang-format -i $(SRC) $(HEADERS)
	
DBG_ARGS = -q --args $(RUN_ARGS)

dbg: edge
	$(DBG) $(DBG_ARGS)

leak_check: edge
	sudo $(VAL) $(VALGRIND_ARGS) $(RUN_ARGS)
