import glob
from PIL import Image


def make_gif(frame_folder, gif_name="my_awesome.gif", images_format='png', save_path='./'):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.{images_format}"))]
    frame_one = frames[0]
    frame_one.save(save_path+gif_name+'.gif', format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)
    print('GIF saved in:\n\t{:s}'.format(save_path+gif_name))


if __name__ == "__main__":
    make_gif("/path/to/images")