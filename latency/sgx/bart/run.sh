# 运行ree统计算子运行耗时
python scripts/ree.py

# 运行tee统计算子运行耗时
make clean && make SGX=1
gramine-sgx ./python scripts/tee.py

# 根据生成耗时文件统计总耗时结果
python scripts/get_results.py