#ifndef USER_TA_HEADER_DEFINES_H
#define USER_TA_HEADER_DEFINES_H

/* Get the TA UUID definition */
#include "include/mnist_demo.h"

#define TA_UUID             TA_MNIST_DEMO_UUID

/*
 * TA properties: multi-instance TA, no specific attribute
 * TA_FLAG_EXEC_DDR is mandated for user TAs
 */
#define TA_FLAGS            TA_FLAG_EXEC_DDR

/* Provisioned stack size */
#define TA_STACK_SIZE       (64 * 1024)

/* Provisioned heap size for TEE_Malloc() and friends */
/* Large enough for MNIST matrices (784*256*4 = ~800KB max per layer) */
#define TA_DATA_SIZE        (2 * 1024 * 1024)

/* The gpd.ta.version property */
#define TA_VERSION          "2.0"

/* The gpd.ta.description property */
#define TA_DESCRIPTION      "MNIST Secure Inference Demo v2 (Parallel Optimized)"

#endif /* USER_TA_HEADER_DEFINES_H */
