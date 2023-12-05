import os
import pickle 


''' Code to get a list of all the files as a sorted list from a specified directory '''
directory_path = 'data/test'
entries = os.listdir(directory_path)
files = sorted([entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))])


''' Impporting the training data as a dictionary '''
file_path = os.path.join(directory_path,files[0])
with open(file_path, 'rb') as file:
    loaded_dict = pickle.load(file)

print(loaded_dict)