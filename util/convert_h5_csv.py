import csv 
import pandas as pd

# data_path="/dataset/NeurIPS2022/train_cite_inputs.h5"

# data = pd.read_hdf(data_path)

# data.to_csv("train_cite_inputs.csv")



# data_path="/dataset/NeurIPS2022/train_cite_targets.h5"

# data = pd.read_hdf(data_path)

# data.to_csv("train_cite_targets.csv")


data_path="/dataset/NeurIPS2022/test_cite_inputs_day_2_donor_27678.h5"

data = pd.read_hdf(data_path)

data.to_csv("test_cite_inputs_day_2_donor_27678.csv")