# We train our GAN model using only images from games released after 2005 and with scaling all training images to a resolution of 128*128 pixels.
# This gives 31255 training images in total.

import pickle
from scipy import misc
import os
import numpy as np
import utils

aim_size = 128
id = 1
year = 2005

dump_filename = './anime_with_tag_new.dat'
tag_path = '../tag_estimation/tags_new.txt'

def get_info_list():
    info_list = []
    with open(tag_path, 'r') as fin:
        info_list = fin.readlines()
    print(info_list)
    info_list = list(map(lambda each: each.strip('\n'), info_list))

    info_list = list(map(lambda each: each.split('\t'), info_list))
    print(info_list)
    return info_list



def process_image(img):
  global id
  # resization
  img = misc.imresize(img, [aim_size, aim_size, 3])
  print('{} finished.'.format(id))
  id += 1
  return img


def dump_file(obj, dump_filename):
  with open(dump_filename, 'wb') as fout:
    pickle.dump(obj, fout)


if __name__ == '__main__':
  info_list = get_info_list()
  result_list = []
  for i, d in enumerate(info_list):
      if os.path.exists(d[3]):
          print(d[1],d[2])
          if int(d[1]) < year:
              continue
          # one_hot = utils.get_one_hot([d[2]])
          result_list.append([utils.get_one_hot([d[2]]), process_image(misc.imread(d[3]))])
  dump_file(result_list, dump_filename)