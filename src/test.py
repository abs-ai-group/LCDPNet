# -*- coding: utf-8 -*-
import time

import hydra
import pytorch_lightning as pl
from omegaconf import open_dict
from pytorch_lightning import Trainer

from globalenv import *
from utils.util import parse_config
import yaml
import os
from pathlib import Path
import shutil

pl.seed_everything(GLOBAL_SEED)


@hydra.main(config_path='config', config_name="config")
def main(opt=None):
    opt = parse_config(opt, TEST)
    print('Running config:', opt)
    from model.lcdpnet import LitModel as ModelClass
    ckpt = opt[CHECKPOINT_PATH]
    assert ckpt
    model = ModelClass.load_from_checkpoint(ckpt, opt=opt)
    # model.opt = opt
    with open_dict(opt):
        model.opt[IMG_DIRPATH] = model.build_test_res_dir()
        opt.mode = 'test'
    print(f'Loading model from: {ckpt}')

    from data.img_dataset import DataModule
    datamodule = DataModule(opt)

    trainer = Trainer(
        gpus=opt[GPU],
        strategy=opt[BACKEND],
        precision=opt[RUNTIME_PRECISION])

    beg = time.time()
    trainer.test(model, datamodule)
    print(f'[ TIMER ] Total time usage: {time.time() - beg}')
    print('[ PATH ] The results are in :')
    print(model.opt[IMG_DIRPATH])


def write_yaml_file(input_paths):
    data = {
        "class": "img_dataset",
        "name": "lcdp_data.test",
        "input": input_paths + "/*",
        "GT": input_paths + "/*"
    }
    with open("config/ds/test.yaml", 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)


def move_images_and_remove_directory(source_dir, destination_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Iterate through files in the source directory
    for filename in os.listdir(source_dir):
        # Check if the file is an image file (you can adjust this condition according to your requirements)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Construct the absolute paths for source and destination files
            source_file = os.path.join(source_dir, filename)
            destination_file = os.path.join(destination_dir, filename)
            # Move the image file to the destination directory
            shutil.move(source_file, destination_file)
    # After moving all image files, remove the source directory
    shutil.rmtree(source_dir)


def process_dataset(dataset_path):
    for split in ["train", "val", "test"]:
        split_path = os.path.join(dataset_path, split)
        image_path = os.path.join(split_path, 'images')
        write_yaml_file(image_path)
        main()
        source_directory = "../pretrained_models/test_result"
        destination_directory = Path(str(image_path).replace("Fisheye_deblur", "Fisheye_contrast"))
        move_images_and_remove_directory(source_directory, destination_directory)


if __name__ == "__main__":
    process_dataset("/srv/aic/Fisheye_deblur")
