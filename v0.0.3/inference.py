import argparse, glob, numpy as np, os, torch, shutil, subprocess, cv2
from os import path as osp
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from desr.utils import VideoReader, VideoWriter, pre_process_batched, batch_enhance_rgb
from tqdm import tqdm
from torchvision.transforms.functional import normalize
from gfpgan import GFPGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.download_util import load_file_from_url


def inference_video(args, video_save_path, device=None, total_workers=1, worker_idx=0):
        # ------------------------ set up background upsampler ------------------------
    if args.bg_upsampler == 'gan-suite':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
            
        else:
            # ---------------------- determine models according to model names ---------------------- #
            args.model_name = args.model_name.split('.pth')[0]
            if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
                file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
            elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
                file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
            elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
                netscale = 4
                file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
            elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                netscale = 2
                file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
            elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
                netscale = 4
                file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
            elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
                netscale = 4
                file_url = [
                    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
                ]

            # ---------------------- determine model paths ---------------------- #
            model_path = os.path.join('weights', args.model_name + '.pth')
            if not os.path.isfile(model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                for url in file_url:
                    # model_path will be updated
                    model_path = load_file_from_url(
                        url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

            # use dni to control the denoise strength
            dni_weight = None
            if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
                wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
                model_path = [model_path, wdn_model_path]
                dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

            # restorer
            bg_upsampler = RealESRGANer(
                scale=netscale,
                model_path=model_path,
                dni_weight=dni_weight,
                model=model,
                tile=args.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True,
            )

            if 'anime' in args.model_name and args.face_enhance:
                print('face_enhance is not supported in anime models, we turned this option off for you. '
                      'if you insist on turning it on, please manually comment the relevant lines of code.')
                args.face_enhance = False
    else:
        bg_upsampler = None

    # ------------------------ set up GFPGAN restorer ------------------------

    if args.face_enhance:  # Use GFPGAN for face enhancement
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth',
            upscale=args.upscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=bg_upsampler)

    else:
        face_enhancer = None


   # ------------------------ set up COLORIZER restorer ------------------------ 
    ''''
    if args.colourize:
        model_path = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_colorization.pth'
        colourizer = ARCH_REGISTRY.get('CodeFormer')(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=['32', '64', '128']).to(device)
        
        ckpt_path = load_file_from_url(url=model_path, 
                                    model_dir='weights/CodeFormer', progress=True, file_name=None)
        checkpoint = torch.load(ckpt_path)['params_ema']
        colourizer.load_state_dict(checkpoint)
        colourizer.eval()

    else:
        colourizer = None
    '''    
    reader = VideoReader(args, total_workers, worker_idx)
    audio = reader.get_audio()
    height, width = reader.get_resolution()
    fps = reader.get_fps()
    writer = VideoWriter(args, audio, height, width, video_save_path, fps)

    pbar = tqdm(total=len(reader), unit='frame', desc='inference')
    if args.batch: #does not support face enhancement
        queue = []
        assert not args.face_enhance
        while True:
            img = reader.get_frame()
            if img is None:
                break
            queue.append(img)
            if len(queue) == args.batches:
                try:
                    output = list(batch_enhance_rgb(upsampler, queue, outscale=args.outscale))
                    queue.clear()
                except RuntimeError as error:
                    print('Error', error)
                    print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
                else:
                    for frame in output:
                        writer.write_frame(frame)
                    pbar.update(args.batch)
                torch.cuda.synchronize(device)
        if len(queue):
            for frame in batch_enhance_rgb(upsampler, queue, outscale=args.outscale):
                writer.write_frame(frame)
                pbar.update(1)
            queue.clear()
            torch.cuda.synchronize(device)
            
        reader.close()
        writer.close()   

    else:
        while True:
            img = reader.get_frame()
            if img is None:
                break
            
            try:    
                if args.face_enhance:     
                    # face enhance and super resolve
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
                writer.write_frame(output)
                
            torch.cuda.synchronize(device)
            pbar.update(1)
            
        reader.close()
        writer.close()
    
def run(args):
    args.video_name = osp.splitext(os.path.basename(args.input))[0]
    video_save_path = osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4')
    
    num_gpus = torch.cuda.device_count()
    num_process = num_gpus * args.num_process_per_gpu
    if num_process == 1:
        inference_video(args, video_save_path)
        return
    
    ctx = torch.multiprocessing.get_context('spawn')
    pool = ctx.Pool(num_process)
    os.makedirs(osp.join(args.output, f'{args.video_name}_out_tmp_videos'), exist_ok=True)
    pbar = tqdm(total=num_process, unit='sub_video', desc='inference')
    for i in range(num_process):
        sub_video_save_path = osp.join(args.output, f'{args.video_name}_out_tmp_videos', f'{i:03d}.mp4')
        pool.apply_async(
            inference_video,
            args=(args, sub_video_save_path, torch.device(i % num_gpus), num_process, i),
            callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()

    # combine sub videos
    # prepare vidlist.txt
    with open(f'{args.output}/{args.video_name}_vidlist.txt', 'w') as f:
        for i in range(num_process):
            f.write(f'file \'{args.video_name}_out_tmp_videos/{i:03d}.mp4\'\n')

    cmd = [
        args.ffmpeg_bin, '-f', 'concat', '-safe', '0', '-i', f'{args.output}/{args.video_name}_vidlist.txt', '-c',
        'copy', f'{video_save_path}'
    ]
    print(' '.join(cmd))
    subprocess.call(cmd)
    shutil.rmtree(osp.join(args.output, f'{args.video_name}_out_tmp_videos'))
    if osp.exists(osp.join(args.output, f'{args.video_name}_inp_tmp_videos')):
        shutil.rmtree(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'))
    os.remove(f'{args.output}/{args.video_name}_vidlist.txt')
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='inputs',
        help='Input video folder. Default: inputs/videos')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x2plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument(
        '--colourize',  action='store_true', help='Colourize Black and white image. Default: False')
    parser.add_argument(
        '-s', '--upscale', type=int, default=2, help='The final upsampling scale of the image. Default: 2')
    parser.add_argument(
        '--bg_upsampler', type=str, default='gan-suite', help='background upsampler. Default: gan-suite')
    parser.add_argument(
        '--bg_tile',
        type=int,
        default=0,
        help='Tile size for background sampler, 0 for no tile during testing. Default: 0')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored video')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
    parser.add_argument('--num_process_per_gpu', type=int, default=1)
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg', help='The path to ffmpeg')
    parser.add_argument('--ffprobe_bin', type=str, default='ffprobe', help='The path to ffprobe')
    parser.add_argument('--batch', action='store_true', help='Batch image processing. Does not support face enhancement')
    parser.add_argument('--batches', type=int, default=4)
    args = parser.parse_args()
    
    
    args.input = args.input.rstrip('/').rstrip('\\')
    os.makedirs(args.output, exist_ok=True)

    run(args)
    

    
if __name__ == '__main__':
    main()


