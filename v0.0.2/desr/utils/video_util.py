import ffmpeg
import sys
import os
from basicsr.utils import imwrite
from os import path as osp
import numpy as np
import glob as glob
import mimetypes
import subprocess
import warnings

def get_video_meta_info(video_path, ffprobe):
    ret = {}
    probe = ffmpeg.probe(video_path, cmd=ffprobe)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    return ret

def get_sub_video(args, num_process, process_idx):
    if num_process == 1:
        return args.input
    meta = get_video_meta_info(args.input, args.ffprobe_bin)
    print(meta)
    duration = int(meta['nb_frames'] / meta['fps'])
    part_time = duration // num_process
    print(f'duration: {duration}, part_time: {part_time}')
    os.makedirs(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'), exist_ok=True)
    out_path = osp.join(args.output, f'{args.video_name}_inp_tmp_videos', f'{process_idx:03d}.mp4')
    cmd = [
        args.ffmpeg_bin, f'-hwaccel cuda -i {args.input}', '-ss', f'{part_time * process_idx}',
        f'-to {part_time * (process_idx + 1)}' if process_idx != num_process - 1 else '', '-async 1', out_path, '-y'
    ]
    print(' '.join(cmd))
    subprocess.call(' '.join(cmd), shell=True)
    return out_path

class VideoReader:
    def __init__(self, args, total_workers=1, worker_idx=0):
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type is None else input_type
        self.audio = None
        self.input_fps = None
        if self.input_type.startswith('video'):
            video_path = get_sub_video(args, total_workers, worker_idx)
            self.stream_reader = (
                ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='bgr24',
                                                loglevel='error').run_async(
                                                    pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
            meta = get_video_meta_info(video_path, args.ffprobe_bin)
            self.width = meta['width']
            self.height = meta['height']
            self.input_fps = meta['fps']
            self.audio = meta['audio']
            self.nb_frames = meta['nb_frames']
        else:
            self.warn = warnings.warn('We do not support this extension file type or directory for restoration')
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

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img
    def get_frame(self):
        if self.input_type.startswith('video'):
            return self.get_frame_from_stream()
        else:
            return self.warn
    
    def close(self):
        if self.input_type.startswith('video'):
            self.stream_reader.stdin.close()
            self.stream_reader.wait()

class VideoWriter:
    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.upscale), int(height * args.upscale)
        if out_height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')

        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 audio,
                                 video_save_path,
                                 pix_fmt='yuv420p',
                                 vcodec='h264_nvenc',
                                 preset = 'fast',
                                 loglevel='error',
                                 acodec='copy').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 video_save_path, pix_fmt='yuv420p', vcodec='h264_nvenc', preset = 'fast',
                                 loglevel='error').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))

    def write_frame(self, frame):
        frame = frame.astype(np.uint8).tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()
        
