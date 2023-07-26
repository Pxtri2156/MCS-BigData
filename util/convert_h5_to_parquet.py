import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

def re_save_h5(old_path, new_path):
    print("Converting into new h5 format")
    data = pd.read_hdf(old_path)
    # print(data.head(5))
    data.to_hdf(new_path, format='table', key='dff')

def convert_hdf5_to_parquet(h5_file, parquet_file, chunksize=100000):
    print("Converting into parquet format")
    stream = pd.read_hdf(h5_file, chunksize=chunksize)
    for i, chunk in enumerate(stream):
        if i == 0:
            # Infer schema and open parquet file on first chunk
            parquet_schema = pa.Table.from_pandas(df=chunk).schema
            parquet_writer = pq.ParquetWriter(parquet_file, parquet_schema, compression='snappy')
        table = pa.Table.from_pandas(chunk, schema=parquet_schema)
        parquet_writer.write_table(table)
    parquet_writer.close()
    
def main():
    # parquet_file="./train_cite_targets.parquet"
    # path='/dataset/NeurIPS2022/train_cite_targets.h5'
    # new_h5_path="./new_train_cite_targets.h5"
    # re_save_h5(path, new_h5_path)
    # convert_hdf5_to_parquet(new_h5_path, parquet_file, chunksize=1000000)

    # parquet_data = pd.read_parquet(parquet_file, engine='pyarrow')
    # print(parquet_data.head(5))

    parquet_path="/workspace/tripx/MCS/big_data/single_cell_data/"
    original_path="/dataset/NeurIPS2022/"
    temp_path = "/workspace/tripx/MCS/big_data/tmp_data/"
    list_files=['train_cite_inputs.h5', 
                'train_cite_targets.h5',
                'test_cite_inputs.h5',
                ]
    for fi in list_files:
        print(f"Processing {fi}")
        # Save new h5 format 
        original_fi =original_path + fi
        new_h5_fi = temp_path + 'new_'+fi
        re_save_h5(original_fi, new_h5_fi)
        # Convert new h5 to parquet
        parquet_fi = parquet_path + fi.replace('.h5', '.parquet')
        convert_hdf5_to_parquet(new_h5_fi, parquet_fi)
        print(f"Converted into {fi.replace('.h5', '.parquet')}")

if __name__ == '__main__':
    main()
    
#### Buggggggggg #################