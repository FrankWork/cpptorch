GCC := g++

OUTPUT := main
SOURCES := $(wildcard *.cc)
CCFLAGS := -std=c++11 -pthread

all: $(OUTPUT)

$(OUTPUT):
	$(GCC) -o $(OUTPUT) $(CCFLAGS) $(SOURCES)

clean:
	rm $(OUTPUT)

.PHONY: all
