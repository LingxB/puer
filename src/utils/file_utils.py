import yaml
from dotenv import load_dotenv, find_dotenv
import os
from src.utils.att_dict import AttributeDict
from src.utils.data_utils import create_symbol_dict


def read_config(config_file, obj_view=True):
    cfg = read_yaml(config_file + '.yml')
    if obj_view:
        cfg = AttributeDict(**cfg)
    return cfg

def save_yaml(dict, path, mode='w'):
    with open(path, mode) as outfile:
        yaml.dump(dict, outfile, default_flow_style=False)

def read_yaml(yaml_file):
    with open(yaml_file, 'r') as ymlfile:
        yml = yaml.load(ymlfile)
    return yml

def get_envar(key):
    """Get environment variable form .env file"""
    load_dotenv(find_dotenv(raise_error_if_not_found=True, usecwd=True))
    return os.environ.get(key)

def list_files(path):
    l = list(os.walk(path))
    if l == []:
        return l
    else:
        return l[0][-1]

def load_symbod_dict_if_exists(path, symbol_dict):
    sd_file = symbol_dict + '.yml'
    config_files = list_files(path)
    if sd_file not in config_files:
        return False
    else:
        sd = read_yaml(path + '/' + sd_file)
        return sd

def create_dump_symbol_dict(corpus, start_idx, dump_path, file_name):
    sd, _ = create_symbol_dict(corpus, start_idx)
    save_yaml(sd, dump_path + '/' + file_name + '.yml', 'x')
    return sd

