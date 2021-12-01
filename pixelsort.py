from PIL import Image
import numpy as np
import multiprocessing as mp

from numpy.lib.index_tricks import diag_indices


CONST_STR_VERTICAL = "vertical"
CONST_STR_HORIZONTAL = "horizontal"
CONST_STR_DIAGONAL = "diagonal"

no_threads = 7
direction = CONST_STR_DIAGONAL
reverse = True
brightness_threshold_bot = 100

def sort_method1(array_):
    brightness_threshold_top = 100

    return_arr = array_

    # select subpixels to sort
    start = -1
    for i, p in enumerate(return_arr):
        if sum(p) > brightness_threshold_top:
            start = i
            break

    # sorting
    if start != -1:
        array = return_arr[start:]
        summed_rbg_values = np.sum(array, axis=1)
        zipped = zip(summed_rbg_values, array)
        sorted_tuple_arr = sorted(zipped, key=lambda tup:tup[0])
        array_sorted = np.array([rgb for _, rgb in sorted_tuple_arr])

        if reverse:
            array_sorted = np.flip(array_sorted, axis=0)
        
        return_arr[start:] = array_sorted
    
    return return_arr

def sort_method2(array_):
    return_arr = array_

    # rgba(107,20,13,255)
    # select subpixels to sort
    start = -1
    start_sort = False
    for i, p in enumerate(return_arr):
        r,g,b = p
        if  80 < r and r < 130 and 0 < g and g < 50 and 0 < b and b < 50:
            start = i
            break

    # sorting
    if start != -1:
        array = return_arr[start:]
        summed_rbg_values = np.sum(array, axis=1)
        zipped = zip(summed_rbg_values, array)
        sorted_tuple_arr = sorted(zipped, key=lambda tup:tup[0])
        array_sorted = np.array([rgb for _, rgb in sorted_tuple_arr])

        if reverse:
            array_sorted = np.flip(array_sorted, axis=0)
        
        return_arr[start:] = array_sorted
    
    return return_arr


def sort(array):
    return sort_method1(array)

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

def write_diagonal1(diag_idx, img, data):
    height, width = img.shape[0:2]
    
    if diag_idx < height:
        col_idx = np.arange(start=height-1-diag_idx, stop=height, step= 1)
        row_idx = np.arange(diag_idx+1)
        img[col_idx, row_idx] = data
    else: # diag_idx >= height
        pass


def write_vertical(col_idx, img, data):
    img[:,col_idx] = data

def write_horizontal(row_idx, img, data):
    img[row_idx] = data

def pixelsort(image_path, direction=CONST_STR_HORIZONTAL):

    image = Image.open(image_path)
    image = np.array(image)
    height, width = image.shape[0:2]

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

    print(new_image_name, height, width)


if __name__ == "__main__":
    img_path = "./mars.jpg"
    pixelsort(img_path, direction)