import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from vqfr.demo_util import VQFR_Demo
from vqfr.utils.video_util import VideoReader, VideoWriter
import imageio as iio2
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method



def inference_video(args, video_input_path, frame_output_path, device=None, total_workers=1, worker_idx=0):
        # ------------------------ set up background upsampler ------------------------
    if args.bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
            
        else:
            args.model_name = args.model_name.split('.')[0]
            if args.model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
            elif args.model_name in ['RealESRGAN_x2plus']:  # x2 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                netscale = 2
                
            # determine model paths
            model_path = os.path.join('experiments/pretrained_models', args.model_name + '.pth')
            if not os.path.isfile(model_path):
                model_path = os.path.join('realesrgan/weights', args.model_name + '.pth')
            if not os.path.isfile(model_path):
                raise ValueError(f'Model {args.model_name} does not exist.')
            
            bg_upsampler = RealESRGANer(
                scale=netscale,
                model_path= model_path,
                model=model,
                tile=args.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # ------------------------ set up VRFR restorer ------------------------
    if args.vqfr_enhance:
        if args.version == '1.0':
            arch = 'v1'
            model_name = 'VQFR_v1-33a1fac5'
            fidelity_ratio = None
        elif args.version == '2.0':
            arch = 'v2'
            model_name = 'VQFR_v2'
            fidelity_ratio = args.fidelity_ratio
            assert fidelity_ratio >= 0.0 and fidelity_ratio <= 1.0, 'fidelity_ratio must in range[0,1]'
        else:
            raise ValueError(f'Wrong model version {args.version}.')
    
        # determine model paths
        model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
        if not os.path.isfile(model_path):
            raise ValueError(f'Model {model_name} does not exist.')
    
        restorer = VQFR_Demo(model_path=model_path, upscale=args.upscale, arch=arch, bg_upsampler=bg_upsampler)

    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth',
            upscale=args.upscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=bg_upsampler)

    else:
        face_enhancer = None
        
    reader = VideoReader(video_input_path)
    fps = reader.get_fps()
    writer = VideoWriter()

    pbar = tqdm(total=len(reader), unit='frame', desc='inference')
    count = 0
    while count < len(reader):
        img = reader.get_frame(count)
        if img is None:
            break
        try:
            if args.vqfr_enhance:
                _, _, output = restorer.enhance(
                img,
                fidelity_ratio=fidelity_ratio,
                has_aligned=args.aligned,
                only_center_face=args.only_center_face,
                paste_back=True)
                
                
            elif args.face_enhance:
                _, _, output = face_enhancer.enhance(
                img,
                has_aligned=args.aligned,
                only_center_face=args.only_center_face,
                paste_back=True)
                
            else:    
                output, _ = bg_upsampler.enhance(img, outscale=args.upscale)
                
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            writer.write(frame_output_path,output, count)
            
        count = count +1    
        pbar.update(1)
    pbar.close()

            
def run(args):
    flag = 1
    if flag  == 1 and torch.cuda.is_available():
        device = torch.device("cuda")
    elif flag == 2 and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    video_input_path, frame_output_path, processes = args.input, args.frame_output, []
    num_gpus = torch.cuda.device_count()
    num_process = num_gpus * args.num_process_per_gpu
    
    if num_process == 1:
        inference_video(args,video_input_path, frame_output_path)
        return
    torch.manual_seed(0)
    set_start_method('spawn', force=True)
    for i in np.arange(num_process):
        # ctx = torch.multiprocessing.get_context('spawn')
        # pool = ctx.Pool(num_process)
        p = mp.Process(
            target=inference_video,
            args=(args, video_input_path, frame_output_path, device, num_process, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    #     pool.apply_async(
    #         inference_video,
    #         args=(args, video_input_path, frame_output_path, torch.device('cuda'), num_process, i),
    #         callback=lambda arg: pbar.update(1))
    # pool.close()
    # pool.join()
    #combine frames to video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='inputs/videos',
        help='Input video folder. Default: inputs/videos')
    parser.add_argument('-fo', '--frame_output', type=str, default='merged_sequence', help='Output folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x2plus',
        help=('Model names: RealESRGAN_x4plus |RealESRGAN_x2plus| RealESRNet_x4plus | Default:RealESRGAN_x2plus'))
    parser.add_argument('-vo', '--video_output', type=str, default='results', help='Video Output folder. Default: results')
    # we use version to select models, which is more user-friendly
    parser.add_argument(
        '-vqfr' , '--vqfr_enhance', action='store_true' , help='use vector_quantized enhancemnt')
    parser.add_argument(
        '-v', '--version', type=str, default='2.0', help='VQFR model version. Option: [1.0, 2.0]. Default: 2.0')
    parser.add_argument(
        '-f',
        '--fidelity_ratio',
        type=float,
        default=0.0,
        help='fidelity range [0,1] in VQFR-v2, 0 for the best quality, 1 for the best fidelity')
    parser.add_argument(
        '-s', '--upscale', type=int, default=2, help='The final upsampling scale of the image. Default: 2')
    parser.add_argument(
        '--bg_upsampler', type=str, default='realesrgan', help='background upsampler. Default: realesrgan')
    parser.add_argument(
        '--bg_tile',
        type=int,
        default=0,
        help='Tile size for background sampler, 0 for no tile during testing. Default: 0')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
    parser.add_argument('--num_process_per_gpu', type=int, default=1)
    args = parser.parse_args()
    
    args.input = args.input.rstrip('/').rstrip('\\')
    os.makedirs(args.frame_output, exist_ok=True)
    os.makedirs(args.video_output, exist_ok=True)
    run(args)
    

    
if __name__ == '__main__':
    main()

#python inference.py -i inputs/videos -fo merged_sequence -vo results --vqfr_enhance -v 2.0 -s 2 -f 0.1 
#python inference.py -i inputs/videos -fo merged_sequence -vo results --face_enhance 
#python inference.py -i inputs/videos -fo merged_sequence -vo results SR
