# auxiliary functions:
import h5py
import numpy as np
import scipy.sparse as sp

# import matlab files in python
def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir   = np.asarray(ds['ir'])
            jc   = np.asarray(ds['jc'])
            out  = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out

path_dataset = 'data/douban_mgcnn/douban/training_test_dataset.mat'

#loading of the required matrices
M = load_matlab_file(path_dataset, 'M')
Otraining = load_matlab_file(path_dataset, 'Otraining')
Otest = load_matlab_file(path_dataset, 'Otest')
Wrow = load_matlab_file(path_dataset, 'W_users') #dense

print('----------------- M -----------------')
print(M)
print(M.shape)
#print('----------------- training -----------------')
#print(Otraining)
#print(Otraining.shape)
#print('----------------- test -----------------')
#print(Otest)
#print(Otest.shape)
#print('----------------- Wrow -----------------')
#print(Wrow)
#print(Wrow.shape)

# Convert the dense matrix to a sparse CSR matrix
sparse_matrix = sp.csr_matrix(M)

print(f"Length of sparse matrix (number of rows): {sparse_matrix.shape[0]}")

# Get the number of non-zero elements
num_non_zero = sparse_matrix.nnz
print(f"Number of non-zero elements: {num_non_zero}")