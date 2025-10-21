CC      ?= $(CROSS_COMPILE)gcc
LD      ?= $(CROSS_COMPILE)ld
AR      ?= $(CROSS_COMPILE)ar
NM      ?= $(CROSS_COMPILE)nm
OBJCOPY ?= $(CROSS_COMPILE)objcopy
OBJDUMP ?= $(CROSS_COMPILE)objdump
READELF ?= $(CROSS_COMPILE)readelf

OBJS = main.o

# Compiler flags (including -fPIC for shared library)
CFLAGS += -Wall -I../ta/include -I$(TEEC_EXPORT)/include -I./include
CFLAGS_SHARED = $(CFLAGS) -fPIC  # Additional flags for shared lib

# Linker flags
LDADD += -lteec -L$(TEEC_EXPORT)/lib

# Targets
BINARY = inference_demo
LIBRARY = libinference.so

.PHONY: all
all: $(LIBRARY) $(BINARY)  # Build both targets

# Rule for shared library
$(LIBRARY): $(OBJS)
	$(CC) -shared $(LDFLAGS) -o $@ $^ $(LDADD)

# Rule for executable
$(BINARY): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDADD)

.PHONY: clean
clean:
	rm -f $(OBJS) $(BINARY) $(LIBRARY)

# Compile rule for .c files (two versions: normal and PIC)
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@  # For executable

shared_objs = $(OBJS:.o=.os)
%.os: %.c
	$(CC) $(CFLAGS_SHARED) -c $< -o $@  # For shared lib (PIC)
