import os
import h5py
import json

def convert_h5_to_json(h5_file_path, json_file_path):
    # Set the HDF5_PLUGIN_PATH environment variable to the plugin directory
    plugin_path = '/workspace/tripx/miniconda3/envs/big_data_v2/lib/python3.10/site-packages/hdf5plugin/plugins/'
    os.environ['HDF5_PLUGIN_PATH'] = plugin_path

    # Open the HDF5 file
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Convert the HDF5 data to a dictionary
        data_dict = {}
        def parse_key(key):
            if isinstance(key, tuple):
                return '/'.join(str(k) for k in key)
            return key

        def convert_item(item):
            if isinstance(item, h5py.Dataset):
                return item[()]
            elif isinstance(item, h5py.Group):
                return convert_group(item)
            else:
                return None

        def convert_group(group):
            group_dict = {}
            for key in group.keys():
                group_dict[parse_key(key)] = convert_item(group[key])
            return group_dict

        data_dict = convert_group(h5_file)

    # Write the dictionary to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

# Provide the paths to the input HDF5 file and the output JSON file
h5_file_path = '/dataset/NeurIPS2022/train_cite_targets.h5'
json_file_path = './train_cite_targets.json'

# Convert the HDF5 file to JSON
convert_h5_to_json(h5_file_path, json_file_path)
