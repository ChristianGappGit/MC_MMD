import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def colorize_img(img_dim, patch_size, token_map, cmap, mapping_range=[0,1]):
    print('img_dim:', img_dim)
    print('patch_size:', patch_size)
    print('token_map_shape:', token_map.shape)
    if len(img_dim) != 2:
        raise ValueError('img_dim must be a list of two numbers')
    if patch_size[0] > img_dim[0] or patch_size[1] > img_dim[1]:
        raise ValueError('patch_size must be smaller than img_dim')
    if img_dim[0] % patch_size[0] != 0 or img_dim[1] % patch_size[1] != 0:
        raise ValueError('img_dim must be divisible by patch_size')
    
    num_patches_dim0 = (img_dim[0]//patch_size[0])
    num_patches_dim1 = (img_dim[1]//patch_size[1])

    if len(token_map) != num_patches_dim0 * num_patches_dim1:
        raise ValueError('token_map must have length equal to the number of patches')
    
    cmap = matplotlib.colormaps.get_cmap(cmap)
    #2D new image
    img = np.zeros((img_dim[0], img_dim[1], 3))
    for i in range(num_patches_dim0):
        for j in range(num_patches_dim1):
            color = cmap(mapping_range[0] + token_map[i*num_patches_dim1 + j]*(mapping_range[1] - mapping_range[0]))
            img[i*patch_size[0]:(i+1)*patch_size[0], j*patch_size[1]:(j+1)*patch_size[1], :] = color[:3]
    return img