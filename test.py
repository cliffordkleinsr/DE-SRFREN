
import cv2
import glob
import numpy as np
import os
import torch
from vqfr.demo_util import VQFR_Demo
from vqfr.utils.video_util import VideoReader, VideoWriter
import imageio as iio2


inp = 'inputs/videos'
save_fldr = 'merged_sequence'
merged = 'results'

reader = VideoReader(inp)
fps =reader.get_fps()

writer = VideoWriter()

writer.merge_frames(merged, save_fldr,fps)
