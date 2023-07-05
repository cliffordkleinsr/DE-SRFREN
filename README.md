# ***DE-SRFREN*** : Video restoration Processing Pipeline
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11m6d16yPfaqauhxKTitb6mBmc4zM3KYS?usp=sharing])

As the name suggests, this is a video restoration pipeline pulling from various cutting edge technologies and merging them to create the one processing pipeline, for videos, to rule them all. The pipeline borrows from multiple  AI techniques from different contributers, these techniques are mentioned in our [releases](https://github.com/cliffordkleinsr/DE-SRFREN/releases) page.
If you like our project please give as a star and also don't forget to like the other projects used by the video restoration pipeline :cowboy_hat_face:

> *NOTE* only one video at a time

# Installation
Setting up the environment
```bash
# Make sure you have git installed
git clone https://github.com/cliffordkleinsr/DE-SRFREN.git
cd DE-SRFREN
# Make sure you have python installed -.-"
# Install basicsr 
pip install basicsr 
# Install facexlib 
# We use face detection and face restoration helper in the facexlib package
pip install facexlib #parsing path net and resnet faces
pip install realesrgan  
pip install gfpgan

pip install -r requirements.txt
python setup.py develop
```
As a side note, make sure you have pytorch compiled with cuda binaries installed otherwise inference speed using the cpu will be greatly impacted

### USAGE
----
- Basic argument structure:
```yaml
-i or --input , your input video directory
-fo or --frame_output , your frame output directory
-vo or --video_output, your video output
-h or --help , for help with arguments
```

- For quick inference with the GFPGAN variant use:
```py
python inference.py -i inputs/videos -fo merged_sequence -vo results --face_enhance --bg_upsampler None #for faster inference
#or
python inference.py -i inputs/videos -fo merged_sequence -vo results --face_enhance #to super resolve your image after restoration
```

- For quick inference with the VQFR variant use:
> **NOTE**: only usable with v0.0.1
```py
python inference.py -i inputs/videos -fo merged_sequence -vo results --vqfr_enhance -v 2.0 -s 2 -f 0.1 --bg_upsampler None #for faster inference
#or
python inference.py -i inputs/videos -fo merged_sequence -vo results --vqfr_enhance -v 2.0 -s 2 -f 0.1 #to super resolve your image after restoration
```
Please [NOTE](https://github.com/cliffordkleinsr/DE-SRFREN/releases#:~:text=I%20realized%20that%20the%20VQFR%20face%20restoration%20pipeline%20does%20not%20work%20well%20with%20a%20subject%27s%20eyes%20and%20hair%20features.) That VQFR has its own caveats
- To use the super-resolution pipeline run:
```py

python inference.py -i inputs/videos -fo merged_sequence -vo results 

```


- To Encode an image sequence to a H.264 MP4 codec run argument:
```py

python test.py #results will be in the results folder

```

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
1. [X] Take a video frame and turn into images
2. [X] Super resolve the image
3. [X] Restore the Faces in each frame step
4. [X] Merge Frames H.264 codec MP4
5. [ ] Speed up inference *(main focus)

Feature Requests
-------------
1. [ ] Frame Generation 24-60 FPS
2. [ ] More support for different video formats
2. [ ] Color BLack and White Images
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
