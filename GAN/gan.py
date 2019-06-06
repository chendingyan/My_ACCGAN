from generator import Generator
from discriminator import Discriminator
from data_loader import DataLoader
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.autograd import Variable, grad
import utils
import random
import os
import torchvision.utils as vutils
import time


__DEBUG__ = True

# have GPU or not.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##########################################
# Load params

data_path = './anime_with_tag.dat'
learning_rate = 0.0002
beta_1 = 0.5
batch_size= 64
lr_update_cycle = 50000
max_epoch = 500
num_workers= 4
noise_size = 128
lambda_adv = 34.0
lambda_gp = 0.5
verbose = True
verbose_T = 100
model_dump_path = './gan_models'

tmp_path= './training_temp'



if __DEBUG__:
  batch_size = 64
  num_workers = 1
#
#
##########################################


def initital_network_weights(element):
  if hasattr(element, 'weight'):
    element.weight.data.normal_(.0, .02)


def adjust_learning_rate(optimizer, iteration):
  lr = learning_rate * (0.1 ** (iteration // lr_update_cycle))
  return lr


class SRGAN():
  def __init__(self):
    self.dataset = DataLoader(data_path=data_path,
                                    transform=transforms.Compose([ToTensor()]))
    self.data_loader = torch.utils.data.DataLoader(self.dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers, drop_last=True)
    checkpoint, checkpoint_name = self.load_checkpoint(model_dump_path)
    if checkpoint == None:
      self.G = Generator().to(device)
      self.D = Discriminator().to(device)
      self.G.apply(initital_network_weights).to(device)
      self.D.apply(initital_network_weights).to(device)
      self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
      self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
      self.epoch = 0
    else:
      self.G = Generator().to(device)
      self.D = Discriminator().to(device)
      self.G.load_state_dict(checkpoint['G'])
      self.D.load_state_dict(checkpoint['D'])
      self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
      self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
      self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
      self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
      self.epoch = checkpoint['epoch']


    self.label_criterion = nn.BCEWithLogitsLoss().to(device)
    self.tag_criterion = nn.MultiLabelSoftMarginLoss().to(device)


  def load_checkpoint(self, model_dir):
    models_path = utils.read_newest_model(model_dir)
    if len(models_path) == 0:
      return None, None
    models_path.sort()
    new_model_path = os.path.join(model_dump_path, models_path[-1])
    if torch.cuda.is_available():
      checkpoint = torch.load(new_model_path)
    else:
      checkpoint = torch.load(new_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    return checkpoint, new_model_path


  def train(self):
    iteration = -1
    label = Variable(torch.FloatTensor(batch_size, 1)).to(device)
    while self.epoch <= max_epoch:
      adjust_learning_rate(self.optimizer_G, iteration)
      adjust_learning_rate(self.optimizer_D, iteration)
      for i, (anime_tag, anime_img) in enumerate(self.data_loader):
        iteration += 1
        if anime_img.shape[0] != batch_size:
          continue
        anime_img = Variable(anime_img).to(device)
        anime_tag = Variable(torch.FloatTensor(anime_tag)).to(device)
        # D : G = 2 : 1
        # 1. Training D
        # 1.1. use real image for discriminating
        self.D.zero_grad()
        label_p, tag_p = self.D(anime_img)
        label.data.fill_(1.0)

        # 1.2. real image's loss
        real_label_loss = self.label_criterion(label_p, label)
        real_tag_loss = self.tag_criterion(tag_p, anime_tag)
        real_loss_sum = real_label_loss * lambda_adv / 2.0 + real_tag_loss * lambda_adv / 2.0
        real_loss_sum.backward()


        # 1.3. use fake image for discriminating
        g_noise, fake_tag = utils.fake_generator(batch_size, noise_size, device)
        fake_feat = torch.cat([g_noise, fake_tag], dim=1)
        fake_img = self.G(fake_feat).detach()
        fake_label_p, fake_tag_p = self.D(fake_img)
        label.data.fill_(.0)

        # 1.4. fake image's loss
        fake_label_loss = self.label_criterion(fake_label_p, label)
        fake_tag_loss = self.tag_criterion(fake_tag_p, fake_tag)
        fake_loss_sum = fake_label_loss * lambda_adv / 2.0 + fake_tag_loss * lambda_adv / 2.0
        fake_loss_sum.backward()

        # 1.5. gradient penalty
        # https://github.com/jfsantos/dragan-pytorch/blob/master/dragan.py
        alpha_size = [1] * anime_img.dim()
        alpha_size[0] = anime_img.size(0)
        alpha = torch.rand(alpha_size).to(device)
        x_hat = Variable(alpha * anime_img.data + (1 - alpha) * \
                         (anime_img.data + 0.5 * anime_img.data.std() * Variable(torch.rand(anime_img.size())).to(device)),
                         requires_grad=True).to(device)
        pred_hat, pred_tag = self.D(x_hat)
        gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).to(device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0].view(x_hat.size(0), -1)
        gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        # gradient_penalty.requires_grad = True
        gradient_penalty = Variable(gradient_penalty, requires_grad = True)
        gradient_penalty.backward()


        # 1.6. update optimizer
        self.optimizer_D.step()

        # 2. Training G
        # 2.1. generate fake image
        self.G.zero_grad()
        g_noise, fake_tag = utils.fake_generator(batch_size, noise_size, device)
        fake_feat = torch.cat([g_noise, fake_tag], dim=1)
        fake_img = self.G(fake_feat)
        fake_label_p, fake_tag_p = self.D(fake_img)
        label.data.fill_(1.0)

        # 2.2. calc loss
        label_loss_g = self.label_criterion(fake_label_p, label)
        tag_loss_g = self.tag_criterion(fake_tag_p, fake_tag)
        loss_g = label_loss_g  * lambda_adv / 2.0 + tag_loss_g * lambda_adv / 2.0
        loss_g.backward()


        # 2.2. update optimizer
        self.optimizer_G.step()


        if iteration % verbose_T == 0:
          print('The iteration is now %d' %iteration)
          print('The loss is %.4f, %.4f, %.4f, %.4f' %(real_loss_sum, fake_loss_sum, gradient_penalty, loss_g ))
          vutils.save_image(anime_img.data.view(batch_size, 3, anime_img.size(2), anime_img.size(3)),
                            os.path.join(tmp_path, 'real_image_{}.png'.format(str(iteration).zfill(8))))
          g_noise, fake_tag = utils.fake_generator(batch_size, noise_size, device)
          fake_feat = torch.cat([g_noise, fake_tag], dim=1)
          fake_img = self.G(fake_feat)
          vutils.save_image(fake_img.data.view(batch_size, 3, anime_img.size(2), anime_img.size(3)),
                            os.path.join(tmp_path, 'fake_image_{}.png'.format(str(iteration).zfill(8))))
      # dump checkpoint
      torch.save({
        'epoch': self.epoch,
        'D': self.D.state_dict(),
        'G': self.G.state_dict(),
        'optimizer_D': self.optimizer_D.state_dict(),
        'optimizer_G': self.optimizer_G.state_dict(),
      }, '{}/checkpoint_{}.tar'.format(model_dump_path, str(self.epoch).zfill(4)))
      self.epoch += 1


if __name__ == '__main__':
  # if not os.path.exists(model_dump_path):
  #   os.mkdir(model_dump_path)
  # if not os.path.exists(tmp_path):
  #   os.mkdir(tmp_path)
  gan = SRGAN()
  # gan.train()
  checkpoint, checkpoint_name = gan.load_checkpoint('./gan_models')
  epoch = checkpoint['epoch']
  print(epoch)
  dis = checkpoint['D']
  gen = checkpoint['G']
  # label_p, tag_p = self.D(anime_img)