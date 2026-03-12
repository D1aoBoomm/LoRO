export V?=0

# OP-TEE paths - adjust these for your environment
export TEEC_EXPORT ?= /usr/lib
export TA_DEV_KIT_DIR ?= $(HOME)/export-ta_arm64
export CFG_TEE_TA_LOG_LEVEL ?= 4

# Cross-compilation (optional)
HOST_CROSS_COMPILE ?= $(CROSS_COMPILE)
TA_CROSS_COMPILE ?= $(CROSS_COMPILE)

.PHONY: all
all:
	$(MAKE) -C host CROSS_COMPILE="$(HOST_CROSS_COMPILE)" --no-builtin-variables
	$(MAKE) -C ta CROSS_COMPILE="$(TA_CROSS_COMPILE)" LDFLAGS=""

.PHONY: clean
clean:
	$(MAKE) -C host clean
	$(MAKE) -C ta clean

.PHONY: install
install:
	sudo cp ./ta/12345678-1234-1234-abcd-0002a5d5c567.ta /usr/lib/optee_armtz/

.PHONY: test
test:
	sudo ./host/mnist_demo
