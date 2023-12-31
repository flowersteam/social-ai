import csv
import os
import torch
import logging
import sys
from pathlib import  Path

import utils


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    if "RL_STORAGE" in os.environ:
        return os.environ["RL_STORAGE"]
    return "storage"


def get_model_dir(model_name):
    return os.path.join(get_storage_dir(), model_name)


def get_status_path(model_dir, num_frames=None):
    if num_frames:
        return os.path.join(model_dir, "status_{}.pt".format(num_frames))
    return os.path.join(model_dir, "status.pt")

def get_model_path(model_dir, num_frames=None):
    if num_frames:
        return os.path.join(model_dir, "model_{}.pt".format(num_frames))
    return os.path.join(model_dir, "model.pt")


def load_status(status_path):
    return torch.load(status_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def get_status(model_dir, num_frames=None):
    path = get_status_path(model_dir, num_frames)
    # return torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return load_status(path)



def save_status(status, model_dir, num_frames=None):
    path = get_status_path(model_dir, num_frames)
    utils.create_folders_if_necessary(path)
    torch.save(status, path)

def save_model(model, model_dir, num_frames=None):
    path = get_model_path(model_dir, num_frames)
    utils.create_folders_if_necessary(path)
    torch.save(model, path)

def load_model(model_name, raise_not_found=True):
    path = get_model_path(model_name)
    try:
        if torch.cuda.is_available():
            model = torch.load(path)
        else:
            model = torch.load(path, map_location=torch.device("cpu"))
        model.eval()
        return model
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No model found at {}".format(path))

def get_vocab(model_dir):
    return get_status(model_dir)["vocab"]


def get_model_state(model_dir):
    return get_status(model_dir)["model_state"]


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()


def get_csv_logger(model_dir):
    csv_path = os.path.join(model_dir, "log.csv")
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)
