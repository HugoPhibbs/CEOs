import os
import pandas as pd 
import h5py

local_path = "/shared/Dataset/ANNS/ANN-Benchmark/glove-100-angular.hdf5"

with h5py.File(local_path, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())

    train_data = f['train'][:]
    test_data = f['test'][:]


dataset_dir = r"/home/hphi344/Documents/CEOs/test/datasets"

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

train_data.tofile(f"{dataset_dir}/glove-100_dataset.bin")
test_data.tofile(f"{dataset_dir}/glove-100_queries.bin")