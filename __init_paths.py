import sys
import os.path as osp

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

WORK_ROOT = '/home/guanjz'
OUT_ROOT = './result'
LOG_ROOT = './log'
TEMP_ROOT ='./temp'
DATA_ROOT = ''
VIDEO_ROOT = ''