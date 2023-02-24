# _DE-SRFREN_
Video restoration Processing Pipeline

> *NOTE* only one video at a time
USAGE
----
- Basic argument structure:
```yaml
-i or --input , your input directory
-o or --output , your output directory
-h or --help , for help with arguments
```
- To Decode a video to an image sequence run argument:
```py
python decode_video_to_frames.py -i Input -o Output
```
- To Encode an image sequence to a H.264 MP4 codec run argument:
```py
python encode_frames_to_video.py - i Input - o Output
```

# Goals
1. [X] Take a video frame and turn into images
2. [ ] Super resolve the image
3. [ ] Restore the Faces in each frame step
4. [X] Merge Frames H.264 codec MP4

Feature Requests
-------------
1. [ ] Frame Generation 24-60 FPS
2. [ ] Color BLack and White Images
3. [ ] TBA 
4. [ ] Lossless Decoding and encoding? TIFF?
5. [ ] Sound restoration *

Question
---------
What AIs to use? [#2](https://github.com/cliffordkleinsr/DE-SRFREN/issues/2#issue-1598488233)

