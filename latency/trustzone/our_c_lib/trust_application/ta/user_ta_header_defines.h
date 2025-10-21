#ifndef USER_TA_HEADER_DEFINES_H
#define USER_TA_HEADER_DEFINES_H

/* To get the TA UUID definition */
#include <demo.h>

#define TA_UUID				TA_DEMO_INFERENCE

/*
 * TA properties: multi-instance TA, no specific attribute
 * TA_FLAG_EXEC_DDR is meaningless but mandated.
 */
#define TA_FLAGS			TA_FLAG_EXEC_DDR

/* Provisioned stack size */
#define TA_STACK_SIZE			(4096 * 4096)

/* Provisioned heap size for TEE_Malloc() and friends */
#define TA_DATA_SIZE			(4096 * 4096)

/* The gpd.ta.version property */
#define TA_VERSION	"1.0"

/* The gpd.ta.description property */
#define TA_DESCRIPTION	"Inference Efficiency in TEE"

#endif /* USER_TA_HEADER_DEFINES_H */
