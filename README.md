# DE-SRFREN
Video Restoration Processing Pipeline

> We've released a public colab notebook for use! use the link below to try:

> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sWBOUMiT_lRO8HGYxW2aQRtixQUnL3Yo?usp=sharing])

As the name suggests, this is a video restoration pipeline pulling from various cutting-edge technologies and merging them to create one processing pipeline, for videos, to rule them all. The pipeline borrows from multiple  AI techniques from different contributors, these techniques are mentioned in our [releases](https://github.com/cliffordkleinsr/DE-SRFREN/releases) page.
If you like our project please give us a star and also don't forget to like the other projects used by the video restoration pipeline :cowboy_hat_face:

> *NOTE* Only one video at a time!

# Installation
Setting up the environment
```bash
# Make sure you have git installed
git clone https://github.com/cliffordkleinsr/DE-SRFREN.git
cd DE-SRFREN/v0.0.2
# Make sure you have Python and PyTorch installed -.-"
# Install basicsr 
pip install basicsr 
# Install facexlib 
# We use face detection and face restoration helper in the facexlib package
pip install facexlib #parsing path net and ResNet faces
pip install realesrgan  
pip install gfpgan
pip install -r requirements.txt
```
As a side note, make sure you have Pytorch compiled with Cuda binaries installed otherwise inference speed will be greatly impacted

### USAGE
----
- Basic argument structure:
```yaml
-i or --input, your input video directory
-o or --video_output, your video output
-n, model name
--ffmpeg_bin, path to ffmpeg.exe
--ffprobe_bin, path to fprobe.exe
-h or --help, for help with arguments
```
**Note** The arguments --ffmpef_bin and --ffprobe_bin should only be used if you have not specified the 'ffmpeg binaries' in your environment variables.

- For quick inference on Windows
```py
python inference.py -i inputs/your_video.mp4 --ffmpeg_bin ffmpeg/bin/ffmpeg.exe --ffprobe_bin ffmpeg/bin/ffprobe.exe --face_enhance --suffix outx2 
```
**Note:** face_enhancer only works with videos of real people, If you are working with anime/animation (cartoon) characters, use:
```py
python inference.py -i inputs/your_anime_video.mp4 --ffmpeg_bin ffmpeg/bin/ffmpeg.exe --ffprobe_bin ffmpeg/bin/ffprobe.exe -n realesr-animevideov3 --suffix outx2
````
- For quick inference on Colab/Linux environment is similar to Windows but avoid using the `--ffmpeg_bin ` and `--ffprobe_bin` when the  binaries are already installed
- The Vector Quantized Code book is deprecated and  thus can only be used with v0.0.1:



# RESULTS
### ORIGINAL
https://user-images.githubusercontent.com/37869706/227707586-bd37fe26-bd15-499c-86d7-3ca1dce8d522.mp4 

### PROCESSED WITH DE-SRFREN
https://user-images.githubusercontent.com/37869706/227707730-c1a1bbed-25e0-422e-9a79-f4b85e5835d1.mp4

Original            |  Processed
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/37869706/228237407-5a06754b-c01d-4b6d-afb6-b7042f3f1678.png) | ![image](https://user-images.githubusercontent.com/37869706/228237129-726bf3a4-d5b8-4835-8333-449e1d759749.png)

Original            |  Processed
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/37869706/228238158-b5a271c3-b3b9-42d9-b14f-d2b28e2ab313.png) | ![image](https://user-images.githubusercontent.com/37869706/228238188-e074f4d9-848a-4caf-8b49-3ae04458818c.png)



# Open tasks
1. [X] Take a video frame and turn it into images
2. [X] Super resolve the image
3. [X] Restore the Faces in each frame step
4. [X] Merge Frames H.264 codec MP4
5. [X] Speed up inference Uses NVENC PIPE: Inference now at: `10fps using RTX 4090 | 7fps using RTX 4080 | 3 fps using RTX 3060`. Tested using these three GPUs!

Feature Requests
-------------
1. [ ] Frame Generation 24-60 FPS
2. [ ] More support for different video formats
2. [ ] Color Black and White Images
3. [X] Lossless Decoding and encoding
4. [X] Sound restoration 

 
 # BibTeX
 ```
 @InProceedings{clifford2023desrfren,
    author = {Clifford Njoroge},
    title = {DE-SRFREN: Video restoration Processing Pipeline},
    date = {2023}
  }
  ```
 # Citation
 Real-ESRGAN
 ```
 @InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
```
VQFR
```
@inproceedings{gu2022vqfr,
  title={VQFR: Blind Face Restoration with Vector-Quantized Dictionary and Parallel Decoder},
  author={Gu, Yuchao and Wang, Xintao and Xie, Liangbin and Dong, Chao and Li, Gen and Shan, Ying and Cheng, Ming-Ming},
  year={2022},
  booktitle={ECCV}
}
```
GFPGAN
```
@InProceedings{wang2021gfpgan,
    author = {Xintao Wang and Yu Li and Honglun Zhang and Ying Shan},
    title = {Towards Real-World Blind Face Restoration with Generative Facial Prior},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2021}
}
```
IMAGEIO

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7751274.svg)](https://doi.org/10.5281/zenodo.7751274)
