'''
The code is modified from the Real-ESRGAN:
https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan_video.py

'''

import sys
import os
from basicsr.utils import imwrite
import numpy as np
import glob as glob
import imageio.v3 as iio
import imageio as iio2


        
    
def get_video_meta_info(video_path):
    ret = {}
    file = glob.glob(video_path +'/*mkv') or glob.glob(video_path +'/*mp4') or glob.glob(video_path+'/*mov') or glob.glob(video_path+'/*avi')
    file = file[0]
    metadata = iio.immeta(file,  plugin='FFMPEG')
    ret['width'] = metadata['size'][0]
    ret['height'] = metadata['size'][1]
    ret['fps'] = metadata['fps']
    ret['audio'] =  None
    ret['nb_frames'] = int(metadata['fps'] * metadata['duration'])
    return ret

class VideoReader:
    def __init__(self, video_path):
        self.audio = None
        file = glob.glob(video_path +'/*mkv') or glob.glob(video_path +'/*mp4') or glob.glob(video_path+'*/mov') or glob.glob(video_path+'/*avi')
        file = file[0]
        self.stream_reader = [frame for frame in iio.imiter(file, plugin="pyav", format="rgb24", thread_type="FRAME")]
            
        meta = get_video_meta_info(video_path)
        self.width = meta['width']
        self.height = meta['height']
        self.input_fps = meta['fps']
        self.audio = meta['audio']
        self.nb_frames = meta['nb_frames']
        

        self.idx = 0

    def get_resolution(self):
        return self.height, self.width
    
    def get_fps(self):
        if self.input_fps is not None:
            return self.input_fps
        else:
            return 24
        
    def get_audio(self):
        return self.audio
    
    def __len__(self):
        return self.nb_frames
    
    def get_frame_from_stream(self, ix):
        img = self.stream_reader[ix]
        return img

    def get_frame(self, ix):
        return self.get_frame_from_stream(ix)
    
class VideoWriter:
    def __init__(self):
        self.fileList = []

    def write(self, save_path, images ,ix):
        basename =  "temp_processed_"
        #save_restore_path = os.path.join(save_path, f'{basename}{ix}.jpg')
        iio.imwrite(f"{save_path}/temp_processed_{ix:03d}.jpg", images)
        #imwrite(images, save_restore_path)
        
    def merge_frames(self, save_path, image_path, fps):
        writer = iio2.get_writer('{}/processed.mp4'.format(save_path), mode = 'I', fps= fps)
        for file in os.listdir(image_path):
            complete_path = os.path.join(image_path ,file)
            self.fileList.append(complete_path)
        

        for im in self.fileList:
            writer.append_data(iio.imread(im))
        writer.close()  
        
        
