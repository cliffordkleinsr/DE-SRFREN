import imageio.v3 as iio
import argparse
import os
import datetime



def read_and_store_metadata(Input):
    metadata = iio.immeta(Input ,exclude_applied=False)
    with open("codec_info/metadata.txt", "w") as txt:
        txt.write(str(metadata['fps']))
    
def read_imageio_pyav(Input, Output):
    for idx, frame in enumerate(
        iio.imiter(Input, plugin="pyav", format="rgb24", thread_type="FRAME")
    ):
        iio.imwrite(f"{Output}/frame{idx:03d}.jpg", frame)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument('-i', '--input', type=str, default='video_folder', help='Input image or folder')
    a.add_argument('-o', '--output', type=str, default='image_sequence', help='Output folder')
    args = a.parse_args()

    if not os.path.exists(args.input):
        raise NotADirectoryError(args.input, "Input Directory not found")
    if not os.path.exists(args.output):
        raise NotADirectoryError(args.output, "Output Directory not found")
    
    for file in os.listdir(args.input):
        if file.endswith(".mp4"):
            read_and_store_metadata(args.input + "/" + file)
            ts = datetime.datetime.now()
            print(ts.strftime("%Y-%m-%d %H:%M:%S"))
            
            try:
                read_imageio_pyav("imageio:{}".format(file), args.output)
                tn = datetime.datetime.now()
                print(tn.strftime("%Y-%m-%d %H:%M:%S"))
                print(len(os.listdir(args.output)), "Frames Extracted")
                
            except ValueError:
                read_imageio_pyav(args.input + "/" + file, args.output)
                
            tn = datetime.datetime.now()
            print(tn.strftime("%Y-%m-%d %H:%M:%S"))
            print(len(os.listdir(args.output)), "Frames Extracted")



        
    
        
