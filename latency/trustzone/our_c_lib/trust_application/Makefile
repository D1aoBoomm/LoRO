export V?=0

# my setting
#export HOST_CROSS_COMPILE=aarch64-linux-gnu-
#export TA_CROSS_COMPILE=aarch64-linux-gnu-
export TEEC_EXPORT=/usr/lib
export TA_DEV_KIT_DIR=~/export-ta_arm64
export CFG_TEE_TA_LOG_LEVEL=4

# If _HOST or _TA specific compilers are not specified, then use CROSS_COMPILE
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
