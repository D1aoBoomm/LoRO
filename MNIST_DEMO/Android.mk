LOCAL_PATH := $(call my-dir)

# Build host application
include $(CLEAR_VARS)
LOCAL_MODULE := mnist_demo
LOCAL_SRC_FILES := host/main.c
LOCAL_C_INCLUDES := $(LOCAL_PATH)/ta/include
LOCAL_SHARED_LIBRARIES := libteec
LOCAL_MODULE_PATH := $(TARGET_OUT_EXECUTABLES)
include $(BUILD_EXECUTABLE)

# Build shared library for Python bindings
include $(CLEAR_VARS)
LOCAL_MODULE := libmnist_inference
LOCAL_SRC_FILES := host/main.c
LOCAL_C_INCLUDES := $(LOCAL_PATH)/ta/include
LOCAL_SHARED_LIBRARIES := libteec
include $(BUILD_SHARED_LIBRARY)

# Build TA
include $(CLEAR_VARS)
LOCAL_MODULE := 12345678-1234-1234-abcd-0002a5d5c567
LOCAL_SRC_FILES := ta/mnist_ta.c
LOCAL_C_INCLUDES := $(LOCAL_PATH)/ta/include
include $(BUILD_TRUSTED_APP)
