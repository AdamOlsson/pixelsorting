from PIL import Image
import numpy as np
import multiprocessing as mp
import json
from mars_sort import sort_method1, sort_method2, sort_method3, sort_method4


CONST_STR_VERTICAL = "v"
CONST_STR_HORIZONTAL = "h"
CONST_STR_DIAGONAL = "d"


with open("./config.json", "r") as f:
    config = json.load(f)

def sort(array):
    return sort_method4(array, config)

def process(array):
    return sort(array)

def chunk_horizontal(array):
    return [*array]

def chunk_vertical(array):
    return [*np.transpose(array, (1,0,2))]

def chunk_diagonal1(array):
    height, width = array.shape[0:2]
    chunks = []
    for offset in range(height-1, 0, -1):
        chunks.append(np.diagonal(array, offset=-offset).T)

    for offset in range(0, width, 1):
        chunks.append(np.diagonal(array, offset=offset).T)

    return chunks

def diag_idx_to_col_idx(diag_idx, img):
    height = img.shape[0]
    return diag_idx - (height-1)

def write_diagonal1(diag_idx, img, data):
    height, _ = img.shape[0:2]
    
    diag_length = data.shape[0]

    if diag_idx < height:
        # From first diag to just before first full length diag
        cow_idx = np.arange(diag_length)
        row_idx = np.arange(start=height-1-diag_idx, stop=height, step=1)
        img[row_idx, cow_idx] = data
    else:
        diag_length = data.shape[0]
        row_idx = np.arange(diag_length)
        col_idx = np.arange(diag_length) + diag_idx_to_col_idx(diag_idx, img)
        img[row_idx, col_idx] = data

def write_vertical(col_idx, img, data):
    img[:,col_idx] = data

def write_horizontal(row_idx, img, data):
    img[row_idx] = data

def pixelsort(image_path):

    direction  = config["direction"]
    no_threads = config["no_threads"]

    image = Image.open(image_path)
    image = np.array(image)
    height, width = image.shape[0:2]

    print(height, width)

    itr_lenghts = {
        CONST_STR_VERTICAL:   width,
        CONST_STR_HORIZONTAL: height,
        CONST_STR_DIAGONAL:   height-1 + 1 + width-1
    }

    chunk_functions = { 
        CONST_STR_HORIZONTAL: chunk_horizontal,
        CONST_STR_VERTICAL:   chunk_vertical,
        CONST_STR_DIAGONAL:   chunk_diagonal1
    }

    write_functions = {
        CONST_STR_HORIZONTAL:write_horizontal,
        CONST_STR_VERTICAL:write_vertical,
        CONST_STR_DIAGONAL: write_diagonal1
    }

    chunk_fn = chunk_functions[direction]
    write_fn = write_functions[direction]
    itr_len = itr_lenghts[direction]

    with mp.Pool(processes=no_threads) as pool:
        itr = pool.imap(process, chunk_fn(image))
        
        for i, res in enumerate(itr):
            print("{}/{}".format(i, itr_len), end='\r')
            write_fn(i, image, res)


    new_image = Image.fromarray(image)
    new_image_name = "img_pixelsorted.jpg"
    new_image.save(new_image_name)

    print(new_image_name)


if __name__ == "__main__":
    img_path = "./mars.jpg"

    pixelsort(img_path)