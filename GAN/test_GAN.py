import torch
from torch.autograd import Variable
import torchvision.utils as vutils
import os
from generator import Generator
from discriminator import Discriminator
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



tmp_path= './training_temp'
model_dump_path = './gan_models'


def load_checkpoint(model_dir):
  models_path = utils.read_newest_model(model_dir)
  if len(models_path) == 0:
    return None, None
  models_path.sort()
  new_model_path = os.path.join(model_dump_path, models_path[-1])
  checkpoint = torch.load(new_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
  print(new_model_path)
  return checkpoint, new_model_path


def generate(G, file_name, tags,D):
  '''
  Generate fake image.
  :param G:
  :param file_name:
  :param tags:
  :return: img's tensor and file path.
  '''
  # g_noise = Variable(torch.FloatTensor(1, 128)).to(device).data.normal_(.0, 1)
  # g_tag = Variable(torch.FloatTensor([utils.get_one_hot(tags)])).to(device)
  g_noise, g_tag = utils.fake_generator(1, 128, device)

  img = G(torch.cat([g_noise, g_tag], dim=1))
  label_p, tag_p = D(img)
  print(label_p)
  print(tag_p)
  vutils.save_image(img.data.view(1, 3, 128, 128),
                    os.path.join(tmp_path, '{}.png'.format(file_name)))
  print('Saved file in {}'.format(os.path.join(tmp_path, '{}.png'.format(file_name))))
  return img.data.view(1, 3, 128, 128), os.path.join(tmp_path, '{}.png'.format(file_name))


if __name__ == '__main__':
  G = Generator().to(device)
  checkpoint, _ = load_checkpoint(model_dump_path)
  G.load_state_dict(checkpoint['G'])
  # print(G)

  D = Discriminator().to(device)
  D.load_state_dict(checkpoint['D'])
  # print(D)
  img, _ = generate(G, 'test', ['glasses'],D)

