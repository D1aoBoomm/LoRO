nvcc data_transfer.cu -o data_transfer
nvcc -Xcompiler -fPIC -shared -o libtransfer.so data_transfer.cu