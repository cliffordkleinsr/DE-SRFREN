import imageio.v2 as iio
import os
import argparse



def merge_image_sequence(Input, Output, fps):
   fileList = []
   for file in os.listdir(Input):
       if file.startswith('frame'):
           complete_path = Input +'/'+ file
           fileList.append(complete_path)

   writer = iio.get_writer('{}/test.mp4'.format(Output), fps=fps)

   for im in fileList:
       writer.append_data(iio.imread(im))
   writer.close()

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument('-i', '--input', type=str, default='Output', help='Input image or folder')
    a.add_argument('-o', '--output', type=str, default='Temp', help='Output folder')
    args = a.parse_args()

    if not os.path.exists(args.input):
        raise NotADirectoryError(args.input, "Input Directory not found")
    if not os.path.exists(args.output):
        raise NotADirectoryError(args.output, "Output Directory not found")
    fp = "codec_info/metadata.txt"
    
    with open(fp, "r") as d:
       fps = d.read()
    
    merge_image_sequence(args.input, args.output, float(fps))
