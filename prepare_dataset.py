# Script permettant de réduire la taille du dataset.
import os
from tqdm import tqdm
import numpy as np
import math
PATH = "./dataset/asl_alphabet_train2"
percentage_to_keep = 0.25
for dirpath, _, filenames in os.walk(PATH):
    if len(filenames) == 0:
        continue
    np.random.shuffle(filenames)
    
    _, tail = os.path.split(dirpath)
    index_fin = math.floor( (1 - percentage_to_keep) * (len(filenames) - 1))
    for i in range(0, index_fin):
        file_to_delete = filenames[i]
        path_to_delete = os.path.join(dirpath,  file_to_delete)
        print(path_to_delete)
        os.remove(path_to_delete)
