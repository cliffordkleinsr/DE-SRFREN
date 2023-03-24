# ***DE-SRFREN***
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11m6d16yPfaqauhxKTitb6mBmc4zM3KYS?usp=sharing])

Video restoration Processing Pipeline

> *NOTE* only one video at a time
USAGE
----
- Basic argument structure:
```yaml
-i or --input , your input directory
-fo or --frame_output , your frame output directory
-vo or --video_output, your video output
-h or --help , for help with arguments
```
- For quick inference with the gfpgan variant use:
```py
python inference.py -i inputs/videos -fo merged_sequence -vo results --face_enhance --bg_upsampler None #for faster inference
```
or
```py
python inference.py -i inputs/videos -fo merged_sequence -vo results --face_enhance #to super resolve your image after restoration
```
- For quick inference with the VQFR variant use [NOTE](https://github.com/cliffordkleinsr/DE-SRFREN/releases#:~:text=I%20realized%20that%20the%20VQFR%20face%20restoration%20pipeline%20does%20not%20work%20well%20with%20a%20subject%27s%20eyes%20and%20hair%20features.):
```py
python inference.py -i inputs/videos -fo merged_sequence -vo results --vqfr_enhance -v 2.0 -s 2 -f 0.1 --bg_upsampler None #for faster inference
```
or
```py
python inference.py -i inputs/videos -fo merged_sequence -vo results --vqfr_enhance -v 2.0 -s 2 -f 0.1 #to super resolve your image after restoration
```

-For Sr instances _TBA_ (more models coming soon) use:
```py
python inference.py -i inputs/videos -fo merged_sequence -vo results 
```
- To Encode an image sequence to a H.264 MP4 codec run argument:
```py
python test.py #results will be in the results folder
```

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
3. [ ] Lossless Decoding and encoding? TIFF?
4. [ ] Sound restoration *

Question
---------
 [#2](https://github.com/cliffordkleinsr/DE-SRFREN/issues/2#issue-1598488233)

