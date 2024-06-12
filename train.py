from __future__ import print_function
from random import seed
from logging import Formatter, StreamHandler, getLogger, FileHandler
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.backends.cudnn
from model import *
from utils.utils import *
from loss_functions import *
from dataset import get_dataset
import sys
import os
import torchvision
from loss_functions import Loss_Functions
#tensorboard --logdir tensorboard --bind_all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
imsize=256
batch=32
epochs = 10000
lr_gen=0.002
lr_dis=0.002
lr_task=0.005
ex_name="GAN_1"
random_seed=True

class Trainer:
    def __init__(self):
        self.imsize=imsize
        self.acc = 0
        self.best_acc = 0
        self.datasets=["SC", "PC"]
        self.batch=batch
        self.train_loader, self.test_loader, self.data_iter = dict(), dict(), dict()
        for dset in self.datasets:
            train_loader, test_loader = get_dataset(dataset=dset, batch=self.batch, imsize=self.imsize)
            self.train_loader[dset] = train_loader
            self.test_loader[dset] = test_loader
        self.nets, self.optims, self.losses = dict(), dict(), dict()
        self.loss_fns = Loss_Functions()
        self.writer = SummaryWriter('./tensorboard/%s' % ex_name)
        self.logger = getLogger()
        self.checkpoint = './checkpoint/%s/%s' % ("clf",ex_name )
        self.step = 0
        self.iter = epochs
        if random_seed:
                self.manualSeed=random.randint(0, 10000)
        else:
            self.manualSeed = 5688
        self.ex=ex_name
        self.logfile ="test"
        self.tensor_freq=50
        self.eval_freq=100
        self.lr_gan=0.0002
        self.weight_decay=0.0001

    def set_default(self):
        torch.backends.cudnn.benchmark = True

        ## Random Seed ##
        print("Random Seed: ", self.manualSeed)
        seed(self.manualSeed)
        torch.manual_seed(self.manualSeed)
        torch.cuda.manual_seed_all(self.manualSeed)

        ## Logger ##
        file_log_handler = FileHandler(self.logfile)
        self.logger.addHandler(file_log_handler)
        stderr_log_handler = StreamHandler(sys.stdout)
        self.logger.addHandler(stderr_log_handler)
        self.logger.setLevel('INFO')
        formatter = Formatter()
        file_log_handler.setFormatter(formatter)
        stderr_log_handler.setFormatter(formatter)

    def save_networks(self):
        checkpoint_dir = os.path.join(self.checkpoint, str(self.step))
        os.makedirs(checkpoint_dir, exist_ok=True)
        for net in self.nets.keys():
            torch.save(self.nets[net].state_dict(), self.checkpoint + '/%d/net%s.pth' % (self.step, net))

    def load_networks(self, step):
        self.step = step
        for net in self.nets.keys():
            self.nets[net].load_state_dict(torch.load(self.checkpoint + '/%d/net%s.pth' % (step, net)))

    def set_networks(self):
        self.nets['G'] = Generator()
        self.nets['D'] = Discriminator()
        self.nets['T'] = Classifier()
        self.nets['P'] = VGG19()
        for net in self.nets.keys():
            self.nets[net].cuda()

    def set_optimizers(self):
        self.optims['G'] = optim.Adam(self.nets['G'].parameters(), lr=self.lr_gan, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.optims['D'] = optim.Adam(self.nets['D'].parameters(), lr=self.lr_gan, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.optims['T'] = optim.Adam(self.nets['T'].parameters(), lr=lr_task, weight_decay=0.0001)

    def set_zero_grad(self):
        for net in self.nets.keys():
            self.nets[net].zero_grad()

    def set_train(self):
        for net in self.nets.keys():
            self.nets[net].train()

    def set_eval(self):
        for net in self.nets.keys():
            self.nets[net].eval()

    def get_batch(self, batch_data_iter):
        batch_data = dict()
        # Instead of using .next() method, use a for loop to iterate through the DataLoader
        for dset in self.datasets:
            try:
                batch_data[dset] = next(iter(batch_data_iter[dset]))
            except StopIteration:
                batch_data_iter[dset] = iter(self.train_loader[dset])
                batch_data[dset] = next(iter(batch_data_iter[dset]))
        return batch_data

    def train_dis(self, imgs):  # Train Discriminators (D)
        imgs_a=imgs[self.datasets[0]]
        imgs_b=imgs[self.datasets[1]]
        self.set_zero_grad()
        fake_B = self.nets['G'](imgs_a)
        pred_real_B = self.nets['D'](imgs_b)
        pred_fake_B = self.nets['D'](fake_B)

        dis_loss = self.loss_fns.dis(pred_real_B, pred_fake_B)

        errD = dis_loss
        errD.backward()

        for net in ['D']:
            self.optims[net].step()

        self.losses['D'] = dis_loss.data.item()

    def train_task(self, imgs, labels):  # Train Task Networks (T)
        self.set_zero_grad()
        task_img=imgs[self.datasets[0]]
        classify=self.nets['G'](task_img)
        output = self.nets['T'](classify)
        errT = self.loss_fns.task(output, labels[self.datasets[0]])
        errT.backward()
        self.optims['T'].step()
        self.losses['T'] = errT.data.item()

    def train_gen(self, imgs, labels):  # Train Generator(G)
        self.set_zero_grad()
        imgs_a=imgs[self.datasets[0]]
        imgs_b=imgs[self.datasets[1]]
        labels_a = labels[self.datasets[0]]

        fake_B = self.nets['G'](imgs_a)
        pred_fake_B = self.nets['D'](fake_B)
        classify_a = self.nets['T'](fake_B)
        perceptual = self.nets['P'](imgs_a)
        perceptual_converted = self.nets['P'](fake_B)
        perceptual_target = self.nets['P'](imgs_b)

        class_loss = self.loss_fns.task(classify_a, labels_a)
        GAN_loss = self.loss_fns.gen(pred_fake_B)
        style_loss = self.loss_fns.swd_loss(perceptual_converted, perceptual_target)
        identity_loss = self.loss_fns.content_loss(perceptual, perceptual_converted)

        errGen = GAN_loss + style_loss + identity_loss + class_loss

        errGen.backward()

        for net in ['G']:
            self.optims[net].step()
        self.losses["G"] = GAN_loss.data.item()
        self.losses["Style"] = style_loss.data.item()
        self.losses["Id"] = identity_loss.data.item()


    def tensor_board_log(self, imgs, labels):
        nrow = 8
        converted_imgs = self.nets["G"](imgs[self.datasets[0]])

        # Input Images & Reconstructed Images
        x = vutils.make_grid(imgs[self.datasets[0]].detach(), normalize=True, scale_each=True, nrow=nrow)
        self.writer.add_image('1_Input_Images/%s' % self.datasets[0], x, self.step)

        # Converted Images
        x = vutils.make_grid(converted_imgs.detach(), normalize=True, scale_each=True, nrow=nrow)
        self.writer.add_image('3_Converted_Images/%s' % self.datasets[0], x, self.step)

        # Losses
        for loss in self.losses.keys():
            self.writer.add_scalar('Losses/%s' % loss, self.losses[loss], self.step)

    def eval(self, datasets):
        target=datasets[1]
        self.set_eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(self.test_loader[target]):
                imgs, labels = imgs.cuda(), labels.cuda()
                pred = self.nets['T'](imgs)
                _, predicted = torch.max(pred.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
                progress_bar(batch_idx, len(self.test_loader[target]), 'Acc: %.3f%% (%d/%d)'
                             % (100. * correct / total, correct, total))
            # Save checkpoint.
            acc = 100. * correct / total
            self.logger.info('======================================================')
            self.logger.info('Step: %d | Acc: %.3f%% (%d/%d)'
                             % (self.step / len(self.test_loader[target]), acc, correct, total))
            self.logger.info('======================================================')
            self.writer.add_scalar('Accuracy/%s' % target, acc, self.step)
            if acc > self.best_acc:
                self.best_acc = acc
                self.writer.add_scalar('Best_Accuracy/%s' % target, acc, self.step)
                self.save_networks()

        self.set_train()

    def print_loss(self):
        best = 'best accuracy' + ': %.2f' % self.best_acc + '|'
        losses = ''
        for key in self.losses:
            losses += ('%s: %.2f|' % (key, self.losses[key]))
        self.logger.info(
            '[%d/%d] %s| %s %s'
            % (self.step, self.iter, losses, best, self.ex))

    def train(self):
        self.set_default()
        self.set_networks()
        self.set_optimizers()
        self.set_train()
        self.logger.info(self.loss_fns.alpha)
        batch_data_iter = dict()
        for dset in self.datasets:
            batch_data_iter[dset] = iter(self.train_loader[dset])

        for i in range(self.iter):
            self.step += 1
            # get batch data
            batch_data = self.get_batch(batch_data_iter)
            imgs, labels = dict(), dict()
            min_batch = self.batch
            for dset in self.datasets:
                imgs[dset], labels[dset] = batch_data[dset]
                imgs[dset], labels[dset] = imgs[dset].cuda(), labels[dset].cuda()
                if imgs[dset].size(0) < min_batch:
                    min_batch = imgs[dset].size(0)
            if min_batch < self.batch:
                for dset in self.datasets:
                    imgs[dset], labels[dset] = imgs[dset][:min_batch], labels[dset][:min_batch]
            # training
            for u in range(1):
                self.train_dis(imgs)
            for t in range(3):
                self.train_gen(imgs, labels)
            self.train_task(imgs, labels)
            # tensorboard
            if self.step % self.tensor_freq == 0:
                self.tensor_board_log(imgs, labels)
            # evaluation
            if self.step % self.eval_freq == 0:
                self.eval(self.datasets)
            self.print_loss()

    def test(self):
        self.set_default()
        self.set_networks()
        self.load_networks(self.load_step)
        self.eval(self.datasets)

trainer = Trainer()
trainer.train()



