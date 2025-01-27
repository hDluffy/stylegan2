import tensorflow as tf
import sys, getopt, os

import numpy as np
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib import tfutil
from dnnlib.tflib.autosummary import autosummary
import math
import numpy as np

from training import dataset
from training import misc
import pickle

from pathlib import Path
import typer
from typing import Optional
import PIL.Image

def tile(X, rows, cols):
    """Tile images for display."""
    tiling = np.zeros((rows * X.shape[1], cols * X.shape[2], X.shape[3]), dtype = X.dtype)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < X.shape[0]:
                img = X[idx,...]
                tiling[
                        i*X.shape[1]:(i+1)*X.shape[1],
                        j*X.shape[2]:(j+1)*X.shape[2],
                        :] = img
    return tiling


def plot_batch(X, rows, cols, out_path):
    """Save batch of images tiled."""
    n_channels = X.shape[3]
    if n_channels > 3:
        X = X[:,:,:,np.random.choice(n_channels, size = 3)]
    canvas = tile(X, rows, cols)
    canvas = np.squeeze(canvas)
    PIL.Image.fromarray(canvas).resize((512*cols,512*rows),PIL.Image.ANTIALIAS).save(out_path)

def extract_conv_names(model):
    # layers are G_synthesis/{res}x{res}/...
    # make a list of (name, resolution, level, position)
    # Currently assuming square(?)

    model_names = list(model.trainables.keys())
    conv_names = []

    resolutions =  [4*2**x for x in range(9)]

    level_names = [["Conv0_up", "Const"],
                    ["Conv1", "ToRGB"]]
    
    position = 0
    # option not to split levels
    for res in resolutions:
        root_name = f"G_synthesis/{res}x{res}/"
        for level, level_suffixes in enumerate(level_names):
            for suffix in level_suffixes:
                search_name = root_name + suffix
                matched_names = [x for x in model_names if x.startswith(search_name)]
                to_add = [(name, f"{res}x{res}", level, position) for name in matched_names]
                conv_names.extend(to_add)
            position += 1

    return conv_names


def blend_models(model_1, model_2, resolution, level, blend_width=None, verbose=False):

    # y is the blending amount which y = 0 means all model 1, y = 1 means all model_2

    # TODO add small x offset for smoother blend animations
    resolution = f"{resolution}x{resolution}"
    
    model_1_names = extract_conv_names(model_1)
    model_2_names = extract_conv_names(model_2)

    assert all((x == y for x, y in zip(model_1_names, model_2_names)))

    model_out = model_1.clone()

    short_names = [(x[1:3]) for x in model_1_names]
    full_names = [(x[0]) for x in model_1_names]
    mid_point_idx = short_names.index((resolution, level))
    mid_point_pos = model_1_names[mid_point_idx][3]
    
    ys = []
    for name, resolution, level, position in model_1_names:
        # low to high (res)
        x = position - mid_point_pos
        if blend_width:
            exponent = -x/blend_width
            y = 1 / (1 + math.exp(exponent))
        else:
            y = 1 if x > 1 else 0

        ys.append(y)
        if verbose:
            print(f"Blending {name} by {y}")

    tfutil.set_vars(
        tfutil.run(
            {model_out.vars[name]: (model_2.vars[name] * y + model_1.vars[name] * (1-y))
             for name, y 
             in zip(full_names, ys)}
        )
    )

    return model_out

def main(low_res_pkl: Path="", # Pickle file from which to take low res layers
         high_res_pkl: Path="", # Pickle file from which to take high res layers
         resolution: int=8, # Resolution level at which to switch between models
         level: int  = 0, # Switch at Conv block 0 or 1?
         generate_num: int=10,
         blend_width: Optional[float] = None, # None = hard switch, float = smooth switch (logistic) with given width
         result_dir: str = "./result/", # Path of image file to save example
         seed: int = 0, # seed for random grid
         output_pkl: Optional[Path] = None, # Output path of pickle (None = don't save)
         verbose: bool = False, # Print out the exact blending fraction
         ):

    tflib.init_tf()
    os.makedirs(result_dir, exist_ok=True)
    resolutions=[512,8,16,32]
    imgs = list()
    with tf.Session() as sess, tf.device('/gpu:0'):
            low_res_G, low_res_D, low_res_Gs = misc.load_pkl(low_res_pkl)
            high_res_G, high_res_D, high_res_Gs = misc.load_pkl(high_res_pkl)
            for i in range(len(resolutions)):
                out = blend_models(low_res_Gs, high_res_Gs, resolutions[i], level, blend_width=blend_width, verbose=verbose)

                rnd = np.random.RandomState(seed)
                for j in range(generate_num): 
                    latents = rnd.randn(1, *out.input_shape[1:])
                    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
                    image = out.run(latents, None, is_validation=True, minibatch_size=1,output_transform=fmt)
                    imgs.append(image[0])
            # Save image.
            imgs = np.stack(imgs, axis = 0)
            print(imgs.shape)
            plot_batch(imgs, len(resolutions), generate_num, os.path.join(result_dir, 'result_show.jpg'))
            
        
if __name__ == '__main__':
    typer.run(main)
    