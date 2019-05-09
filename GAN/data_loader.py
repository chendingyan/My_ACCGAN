import os
from torch.utils.data import Dataset
import PIL.Image as Image
import pickle


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

if __name__ == '__main__':
    dl = DataLoader('./anime_with_tag.dat')
    print(dl.__getitem__(2))