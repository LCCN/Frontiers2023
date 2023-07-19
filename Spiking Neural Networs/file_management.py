import numpy as np
from copy import deepcopy
import os
from pathlib import Path
import pickle
import bz2
import lzma
import json
import zipfile

import sys
sys.stdout.flush()
try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell': # script being run in Jupyter notebook
        from tqdm.notebook import tqdm
    elif shell == 'TerminalInteractiveShell': #script being run in iPython terminal
        from tqdm import tqdm
except NameError:
    if sys.stderr.isatty():
        from tqdm import tqdm
    else:
        from tqdm import tqdm # Probably runing on standard python terminal. If does not work => should be replaced by tqdm(x) = identity(x)

parent_dir =  '/home/jorgemedina/Documents/Doctorado/Animal_mobility/utils/data'
#os.path.join(os.getcwd(), 'data')
##############################################################################################################################
"""                                                  I. Move / delete                                                      """
##############################################################################################################################

def move_files(keyword, folder=None, parent_dir=parent_dir):
    """
    Moves all files in the parent directory starting with a keyword to a folder.
    Attributes:
    - keyword: keyword to look for during file matching-
    - folder: Created folder to store all files matched by the keyword
    - parend_dir: Initial directory in which the files are contained.
    """
    folder = folder if folder is not None else keyword
    Path(os.path.join(parent_dir, folder)).mkdir(exist_ok=True, parents=True)
    processed = 0
    for path in os.listdir(parent_dir):
        if path == folder:
            continue
        else:
            string_list = [string.split('_') for string in path.split('-')]
            string_list_flatten = [item for sublist in string_list for item in sublist]
            if keyword in string_list_flatten:
                os.rename(os.path.join(parent_dir, path), os.path.join(parent_dir, folder, path))
                processed += 1
    return processed

def delete_stdin_files(parent_dir="nuredduna_programmes/stdin_files", extensions=[".e", ".o"]):
    """
    Removes files in parent_dir with extensions starting by (or being equal to) a certain character.
    Originally created to delete nuredduna standard input (stdin) files, of the form python.exxxx (s. error) and python.oxxxx (s. output).
    """
    deleted = 0
    for path in os.listdir(parent_dir):
        for extension in extensions:
            if os.path.splitext(path)[1].startswith(extension):
                os.remove(os.path.join(parent_dir, path))
                deleted += 1
    return deleted


def empty_trash(binDir='/home/jorgemedina/.local/share/Trash'):
    deleted = 0
    for root, dirs, files in os.walk(binDir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
            deleted += 1
    return deleted


##############################################################################################################################
"""                                                      II. Save                                                          """
##############################################################################################################################

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def save_preprocess(filename, saving_type, parent_dir):
    """Creates parent dir if does not exist and ensures filename has the saving_type extension."""
    Path(parent_dir).mkdir(parents=True, exist_ok=True)
    filename = filename if filename.endswith('.{}'.format(saving_type)) else '{}.{}'.format(filename, saving_type)
    return filename
        
def create_saving_func(file_opener, file_writer, saving_type, **kwargs):
    def saving_func(file, filename, parent_dir=parent_dir):
        filename = save_preprocess(filename, saving_type, parent_dir)
        with file_opener(os.path.join(parent_dir, filename), 'w') as f:
            file_writer(file, f, **kwargs)
        return
    return saving_func

save_json = create_saving_func(open, json.dump, 'json', cls=NpEncoder)
save_bz2 = create_saving_func(bz2.BZ2File, pickle.dump, 'pbz2')
save_lzma = create_saving_func(lzma.LZMAFile, pickle.dump, 'lzma')
        
##############################################################################################################################
"""                                                      III. Load                                                         """
##############################################################################################################################

def create_loading_func(file_opener, file_loader, extra_processing=None, apply_defaults=None):
    """
    - extra_processing = List of process that could be applied to the file.
    - apply_defaults = dict:{str: bool}. The key is set as a function variable, the value indicates whether to apply the process named by the key.
    """
    if extra_processing is None:
        def loading_func(path):
            with file_opener(path, 'rb') as f:
                return file_loader(f)
    else: 
        return None
        def loading_func(path):
            args = {key:val for key,val in locals().items() if key not in  ('path', *create_loading_func.__code__.co_varnames)}
            with file_opener(path, 'rb') as f:
                data = file_loader(f)
            
            for condition, process in zip(args.values(), extra_processing):
                if condition:
                    data = process(data)
            return data
        
        # I know this sould not be done this way, but I wanted to check it can
        code_data = loading_func.__code__
        num_vars = code_data.co_argcount
        num_vars_new = len(apply_defaults)
        new_code = code_data.replace(co_varnames=(*code_data.co_varnames[:num_vars], *apply_defaults.keys(), *code_data.co_varnames[num_vars:]),
                                     co_argcount=num_vars + num_vars_new, 
                                     co_nlocals=code_data.co_nlocals + num_vars_new)
        loading_func.__code__ = new_code
        loading_func.__defaults__ = tuple(apply_defaults.values())
    return loading_func

def int_keys(dictionary):
     return {int(key):val for key,val in dictionary.items()}

def load_npz(path, key=None):
    data = np.load(path)
    key = [*data.keys()][0] if key is None else key
    return data[key]

load_json = create_loading_func(open, json.load, extra_processing=[int_keys], apply_defaults={'int_keys':True})
load_bz2 = create_loading_func(bz2.BZ2File, pickle.load)
load_lzma = create_loading_func(lzma.LZMAFile, pickle.load)

##############################################################################################################################
"""                                                   IV. Compression                                                      """
##############################################################################################################################

def compress_files(rootDir=parent_dir, extension='.json', compress_to='lzma', min_size=0, loading_key=None):
    """
    Searches in a directory and all its subdirectories files with a certain extension and compresses them.
    
    Attributes:
    - rootDir: The root directory.
    - extension: Extension setting which files should be compressed. Available options: '.json', '.npz'.
    - compress_to: File type after compression. Available options: 'lzma', 'bz2' (or pbz2).
    - min_size: Minimum size for applying compression (in MB).
    - loading_key: Key for retrieving the data in a npz file. If None, retrieves the data corresponding to the first key.
    
    Returns: Dict containing the name of the files that could not be processed ('bad compression') or those that were corrupted and had to be deleted('deleted files').
    """
    # Get all the paths with the desired extension and minimum size.
    files = []

    for dirpath, subdirList, fileList in os.walk(rootDir):
        files += [os.path.join(dirpath, file) for file in fileList if os.path.splitext(file)[1] == extension]
    
    if min_size > 0:
        files = [file for file in files if os.path.getsize(file) > min_size*1e6]
    
    # Load files, compress, save and delete if compressed_file == pre_compressed_file
    if extension == '.json':
        loader = load_json
    elif extension == '.npz':
        loader = load_npz
    else:
        raise ValueError("extension = '{}' not valid. Available options: '.json', '.npz'".format(extension))
        
    if compress_to == 'lzma':
        compressor = save_lzma
        load_compressor = load_lzma
    elif compress_to == 'bz2':
        compressor = save_bz2
        load_compressor = load_bz2
    else:
        raise ValueError("compress_to = '{}' not valid. Available options: 'lzma', 'bz2'".format(extension))
    
    not_correctly_processed = {'bad_compression': []}
    pbar = tqdm(range(len(files)))
    
    if extension == '.json':
        for file in files:
            try:
                pre_compressed = loader(file, int_keys=True)
            except ValueError:
                pre_compressed = loader(file, int_keys=False)
            
            new_filename = '{}.{}'.format(os.path.splitext(file)[0], compress_to)
            compressor(pre_compressed, new_filename, parent_dir="")
            compressed_file = load_compressor(new_filename)
            
            if compressed_file == pre_compressed:
                os.remove(file)
            else:
                not_correctly_processed['bad_compression'].append(file)
            
            pbar.refresh()
            print(pbar.update())
            
    if extension == '.npz':
        not_correctly_processed['deleted_files'] = []
        for file in files:
            try:
                pre_compressed = loader(file, key=loading_key)
            except zipfile.BadZipFile: # corrupted file
                os.remove(file)
                not_correctly_processed['deleted_files'].append(file)
                continue
                
            new_filename = '{}.{}'.format(os.path.splitext(file)[0], compress_to)
            compressor(pre_compressed, new_filename, parent_dir="")
            compressed_file = load_compressor(new_filename)
            
            if np.all(compressed_file == pre_compressed):
                os.remove(file)
            else:
                not_correctly_processed['bad_compression'].append(file)
            pbar.refresh()
            print(pbar.update())
    
    return not_correctly_processed
        
