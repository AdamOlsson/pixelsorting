from PIL import Image
import numpy as np
import multiprocessing as mp

CONST_STR_VERTICAL = "vertical"
CONST_STR_HORIZONTAL = "horizontal"

no_threads = 7
direction = CONST_STR_VERTICAL
reverse = False

brightness_threshold_top = 500
brightness_threshold_bot = 100

def sort_method1(array):
    summed_rbg_values = np.sum(array, axis=1)
    zipped = zip(summed_rbg_values, array)
    sorted_tuple_arr = sorted(zipped, key=lambda tup:tup[0])
    return np.array([rgb for _, rgb in sorted_tuple_arr])

def extract_method1(array):
    for i, p in enumerate(array):
        if sum(p) > brightness_threshold_top:
            return i, array[i:]
    return None, None


def extract_subarray_to_sort(array):
    return extract_method1(array)

def sort(array):
    return sort_method1(array)

def process(arr):
    start_idx, subarray = extract_subarray_to_sort(arr)

    if start_idx != None:
        arr[start_idx:] = np.flip(sort(subarray), axis=0) if reverse else sort(subarray)

    return arr

def chunk_horizontal(array):
    return [*array]

def chunk_vertical(array):
    return [*np.transpose(array, (1,0,2))]

def write_vertical(col_idx, img, data):
    img[:,col_idx] = data

def write_horizontal(row_idx, img, data):
    img[row_idx] = data

def pixelsort(image_path, direction=CONST_STR_HORIZONTAL):

    image = Image.open(image_path)
    image = np.array(image)
    height, width = image.shape[0:2]

    itr_lenghts = {
        CONST_STR_VERTICAL: width,
        CONST_STR_HORIZONTAL: height
    }

    chunk_functions = { 
        CONST_STR_HORIZONTAL:chunk_horizontal,
        CONST_STR_VERTICAL:chunk_vertical
    }

    write_functions = {
        CONST_STR_HORIZONTAL:write_horizontal,
        CONST_STR_VERTICAL:write_vertical
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
    img_path = "./img.jpg"
    pixelsort(img_path, direction)