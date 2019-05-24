import os
from torch.utils.data import Dataset
import PIL.Image as Image
import pickle
import cv2

tag_list = ['blonde hair','brown hair','black hair','blue hair','pink hair',
               'purple hair','green hair','red hair','silver hair','white hair','orange hair',
               'aqua hair','grey hair','long hair','short hair','twintails','drill hair','ponytail','blush',
               'smile','open mouth','hat','ribbon','glasses','blue eyes','red eyes','brown eyes',
               'green eyes','purple eyes','yellow eyes','pink eyes','aqua eyes','black eyes','orange eyes',]

class DataLoader(Dataset):
  def __init__(self, data_path, valid_years=2005, transform=None, target_transform=None):
    # tag's one-hot, image-bytes
    self.list_pickle = pickle.load(open(data_path, 'rb'))
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, index):
    tag_one_hot = self.list_pickle[index][0]
    image = self.list_pickle[index][1]
    if self.transform is not None:
      image = self.transform(image)
    return tag_one_hot.astype('float32'), image

  def __len__(self):
      return len(self.list_pickle)

def one_hot2tag(tag):
    for i in range(len(tag)):
        if tag[i] == 1:
            return tag_list[i]

if __name__ == '__main__':
    dl = DataLoader('./anime_with_tag.dat')
    # print(dl.__getitem__(2))
    tag, image = dl.__getitem__(1000)
    print(tag)
    tag = one_hot2tag(tag)
    print(tag)
    image = image[:, :, ::-1]
    cv2.namedWindow("Image")
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    #remember to stop programme
