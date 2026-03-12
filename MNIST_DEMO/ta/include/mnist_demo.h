#ifndef TA_MNIST_DEMO_H
#define TA_MNIST_DEMO_H

/*
 * UUID for MNIST Demo v2 Trusted Application (Optimized)
 * Generated with uuidgen
 */
#define TA_MNIST_DEMO_UUID \
	{ 0x12345678, 0x1234, 0x1234, \
		{ 0xab, 0xcd, 0x00, 0x02, 0xa5, 0xd5, 0xc5, 0x67} }

/* Command IDs implemented in this TA */
#define TA_MNIST_CMD_LORO_INFERENCE     0   /* LoRO deobfuscation using stored keys */
#define TA_MNIST_CMD_GENERATE_KEYS      1   /* Generate random keys and store */
#define TA_MNIST_CMD_GET_KEY_STATUS     2   /* Check if keys are loaded */
#define TA_MNIST_CMD_EXPORT_KEYS        3   /* Export keys (for model owner during provisioning) */
#define TA_MNIST_CMD_LOCK_KEYS          4   /* Lock keys after provisioning */

/* Layer dimensions for MNIST 3-layer model */
#define MNIST_INPUT_SIZE    784
#define MNIST_LAYER1_OUT    256
#define MNIST_LAYER2_OUT    128
#define MNIST_OUTPUT_SIZE   10

/* Number of layers */
#define NUM_LAYERS          3

/* Maximum supported dimensions */
#define MAX_BATCH_SIZE      128
#define MAX_INPUT_DIM       1024
#define MAX_OUTPUT_DIM      512
#define MAX_RANK            32

/* Secure storage object ID for LoRO keys */
#define LORO_KEYS_STORAGE_ID    0x1000

/* Key file structure for persistence */
struct loro_key_header {
    uint32_t magic;             /* Magic number for validation */
    uint32_t version;           /* Key format version */
    uint32_t num_layers;        /* Number of layers */
    uint32_t keys_locked;       /* 1 if keys are locked (no export) */
};

struct layer_key_info {
    uint32_t input_dim;         /* Input dimension */
    uint32_t output_dim;        /* Output dimension */
    uint32_t rank;              /* Low-rank dimension */
    uint32_t key_size;          /* Size of B+A matrices in bytes */
};

/* Magic number for key validation */
#define LORO_KEY_MAGIC      0x4C4F524F  /* "LORO" in hex */

#endif /* TA_MNIST_DEMO_H */
