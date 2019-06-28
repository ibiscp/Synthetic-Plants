import math
import os
import timeit
import math

import numpy as np
import ot
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models
from scipy import linalg
import cv2
import time


class Score_knn:
    acc = 0
    acc_real = 0
    acc_fake = 0
    precision = 0
    recall = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0

class Score:
    emd = None
    mmd = None
    knn = None
    inception = None
    mode = None
    fid = None

def load_data(folder_name):
    # Load the dataset
    # files = os.listdir('../dataset/test/')
    files = os.listdir(folder_name + '0/')
    number_files = len(files)
    # print('Number of files: ', number_files)

    X_train = []
    for file in files[0:10]:
        img = cv2.imread('../dataset/test/' + file, 0)
        img = cv2.resize(img, (299, 299))
        X_train.append(img)

    X_train = np.asarray(X_train, dtype='uint8')

    # Rescale
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)
    X_train = np.repeat(X_train, 3, 3)

    return X_train


class pytorchMetrics(object):
    def __init__(self, model='inception_v3'):
        '''
        model: inception_v3, vgg13, vgg16, vgg19, resnet18, resnet34,
               resnet50, resnet101, or resnet152
        '''
        self.model = model
        if self.model.find('vgg') >= 0:
            self.vgg = getattr(models, model)(pretrained=True).cuda().eval()
            self.trans = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        elif self.model.find('resnet') >= 0:
            resnet = getattr(models, model)(pretrained=True)
            resnet.cuda().eval()
            resnet_feature = nn.Sequential(resnet.conv1, resnet.bn1,
                                           resnet.relu,
                                           resnet.maxpool, resnet.layer1,
                                           resnet.layer2, resnet.layer3,
                                           resnet.layer4).cpu().eval()
            self.resnet = resnet
            self.resnet_feature = resnet_feature
            self.trans = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        elif self.model == 'inception' or self.model == 'inception_v3':
            inception = models.inception_v3(
                pretrained=True, transform_input=False).cpu().eval()
            inception_feature = nn.Sequential(inception.Conv2d_1a_3x3,
                                              inception.Conv2d_2a_3x3,
                                              inception.Conv2d_2b_3x3,
                                              nn.MaxPool2d(3, 2),
                                              inception.Conv2d_3b_1x1,
                                              inception.Conv2d_4a_3x3,
                                              nn.MaxPool2d(3, 2),
                                              inception.Mixed_5b,
                                              inception.Mixed_5c,
                                              inception.Mixed_5d,
                                              inception.Mixed_6a,
                                              inception.Mixed_6b,
                                              inception.Mixed_6c,
                                              inception.Mixed_6d,
                                              inception.Mixed_7a,
                                              inception.Mixed_7b,
                                              inception.Mixed_7c,
                                              ).cpu().eval()
            self.inception = inception
            self.inception_feature = inception_feature
            self.trans = transforms.Compose([
                transforms.Resize(299),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            raise NotImplementedError

    def extract(self, images):

        # print('extracting features...')

        feature_pixl, feature_conv, feature_smax, feature_logit = [], [], [], []


        with torch.no_grad():
            input = images.cpu()
            if self.model == 'vgg' or self.model == 'vgg16':
                fconv = self.vgg.features(input).view(input.size(0), -1)
                flogit = self.vgg.classifier(fconv)
                # flogit = self.vgg.logitifier(fconv)
            elif self.model.find('resnet') >= 0:
                fconv = self.resnet_feature(
                    input).mean(3).mean(2).squeeze()
                flogit = self.resnet.fc(fconv)
            elif self.model == 'inception' or self.model == 'inception_v3':
                fconv = self.inception_feature(
                    input).mean(3).mean(2).squeeze()
                flogit = self.inception.fc(fconv)
            else:
                raise NotImplementedError
            fsmax = F.softmax(flogit, dim=1)
            feature_pixl.append(images)
            feature_conv.append(fconv.data.cpu())
            feature_logit.append(flogit.data.cpu())
            feature_smax.append(fsmax.data.cpu())

        feature_pixl = torch.cat(feature_pixl, 0).to('cpu')
        feature_conv = torch.cat(feature_conv, 0).to('cpu')
        feature_logit = torch.cat(feature_logit, 0).to('cpu')
        feature_smax = torch.cat(feature_smax, 0).to('cpu')

        return feature_pixl, feature_conv, feature_logit, feature_smax


    def distance(self, X, Y, sqrt):
        nX = X.size(0)
        nY = Y.size(0)
        X = X.view(nX,-1)
        X2 = (X*X).sum(1).resize_(nX,1)
        Y = Y.view(nY,-1)
        Y2 = (Y*Y).sum(1).resize_(nY,1)

        M = torch.zeros(nX, nY)
        M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) -
                2 * torch.mm(X, Y.transpose(0, 1)))

        del X, X2, Y, Y2

        if sqrt:
            M = ((M + M.abs()) / 2).sqrt()

        return M


    def wasserstein(self, M, sqrt):
        if sqrt:
            M = M.abs().sqrt()
        emd = ot.emd2([], [], M.numpy())

        return emd


    def knn(self, Mxx, Mxy, Myy, k, sqrt):
        n0 = Mxx.size(0)
        n1 = Myy.size(0)
        label = torch.cat((torch.ones(n0), torch.zeros(n1)))
        M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
        if sqrt:
            M = M.abs().sqrt()
        INFINITY = float('inf')
        val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1))).topk(k, 0, False)

        count = torch.zeros(n0 + n1)
        for i in range(0, k):
            count = count + label.index_select(0, idx[i])

        pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1)).float()

        s = Score_knn()
        s.tp = (pred * label).sum()
        s.fp = (pred * (1 - label)).sum()
        s.fn = ((1 - pred) * label).sum()
        s.tn = ((1 - pred) * (1 - label)).sum()
        s.precision = s.tp / (s.tp + s.fp + 1e-10)
        s.recall = s.tp / (s.tp + s.fn + 1e-10)
        s.acc_real = s.tp / (s.tp + s.fn)
        s.acc_fake = s.tn / (s.tn + s.fp)
        s.acc = float(torch.eq(label, pred).float().mean())
        s.k = k

        return s


    def mmd(self, Mxx, Mxy, Myy, sigma):
        scale = Mxx.mean()
        Mxx = torch.exp(-Mxx / (scale * 2 * sigma * sigma))
        Mxy = torch.exp(-Mxy / (scale * 2 * sigma * sigma))
        Myy = torch.exp(-Myy / (scale * 2 * sigma * sigma))
        mmd = math.sqrt(Mxx.mean() + Myy.mean() - 2 * Mxy.mean())

        return mmd


    def entropy_score(self, X, Y, epsilons):
        Mxy = self.distance(X, Y, False)
        scores = []
        for epsilon in epsilons:
            scores.append(self.ent(Mxy.t(), epsilon))

        return scores


    def ent(self, M, epsilon):
        n0 = M.size(0)
        n1 = M.size(1)
        neighbors = M.lt(epsilon).float()
        sums = neighbors.sum(0).repeat(n0, 1)
        sums[sums.eq(0)] = 1
        neighbors = neighbors.div(sums)
        probs = neighbors.sum(1) / n1
        rem = 1 - probs.sum()
        if rem < 0:
            rem = 0
        probs = torch.cat((probs, rem*torch.ones(1)), 0)
        e = {}
        e['probs'] = probs
        probs = probs[probs.gt(0)]
        e['ent'] = -probs.mul(probs.log()).sum()

        return e

    def inception_score(self, X, eps=1e-20):
        kl = X * ((X+eps).log()-(X.mean(0)+eps).log().expand_as(X))
        score = np.exp(kl.sum(1).mean())

        return float(score)

    def mode_score(self, X, Y, eps=1e-20):
        kl1 = X * ((X+eps).log()-(X.mean(0)+eps).log().expand_as(X))
        kl2 = X.mean(0) * ((X.mean(0)+eps).log()-(Y.mean(0)+eps).log())
        score = np.exp(kl1.sum(1).mean() - kl2.sum())

        return float(score)

    def fid(self, X, Y):
        m = X.mean(0)
        m_w = Y.mean(0)
        X_np = X.numpy()
        Y_np = Y.numpy()

        C = np.cov(X_np.transpose())
        C_w = np.cov(Y_np.transpose())
        C_C_w_sqrt = linalg.sqrtm(C.dot(C_w), True).real

        score = m.dot(m) + m_w.dot(m_w) - 2 * m_w.dot(m) + \
            np.trace(C + C_w - 2 * C_C_w_sqrt)
        return float(np.sqrt(score))


    def compute_score(self, real, fake, k=1, sigma=1, sqrt=True):

        # start = time.time()

        # Convert to torch
        x_real = torch.from_numpy(real).permute(0,3,1,2).float()
        x_fake = torch.from_numpy(fake).permute(0,3,1,2).float()

        feature_r = self.extract(x_real)
        feature_f = self.extract(x_fake)

        # Calculate distance
        Mxx = self.distance(feature_r[1], feature_r[1], False)
        Mxy = self.distance(feature_r[1], feature_f[1], False)
        Myy = self.distance(feature_f[1], feature_f[1], False)

        s = Score()
        s.emd = self.wasserstein(Mxy, sqrt)
        s.mmd = self.mmd(Mxx, Mxy, Myy, sigma)
        s.knn = self.knn(Mxx, Mxy, Myy, k, sqrt).acc
        s.inception = self.inception_score(feature_f[3])
        s.mode = self.mode_score(feature_r[3], feature_f[3])
        s.fid = self.fid(feature_r[3], feature_f[3])

        # print("Time:", time.time() - start)

        return s

    def compute_score_raw(self, x_real, x_fake, conv_model='inception_v3', workers=4):

        # Test to make sure implementation is correct
        # np.random.seed(0)
        # feature_r = [torch.from_numpy(np.random.rand(5, 299, 299, 3)).float().cpu(),
        #              torch.from_numpy(np.random.rand(5, 8, 8, 2048)).float().cpu(),
        #              torch.from_numpy(np.random.rand(5, 2048)).float().cpu(),
        #              torch.from_numpy(np.random.rand(5, 1000)).float().cpu()]
        # feature_f = [torch.from_numpy(np.random.rand(5, 299, 299, 3)).float().cpu(),
        #              torch.from_numpy(np.random.rand(5, 8, 8, 2048)).float().cpu(),
        #              torch.from_numpy(np.random.rand(5, 2048)).float().cpu(),
        #              torch.from_numpy(np.random.rand(5, 1000)).float().cpu()]

        # Convert to torch
        x_real = torch.from_numpy(x_real).permute(0,3,1,2).float()
        x_fake = torch.from_numpy(x_fake).permute(0,3,1,2).float()

        feature_r = self.featuresExtractor.extract(x_real)
        feature_f = self.featuresExtractor.extract(x_fake)

        # 4 feature spaces and 7 scores + incep + modescore + fid
        score = np.zeros(4 * 7 + 3)
        for i in range(0, 4):
            #print('compute score in space: ' + str(i))
            Mxx = self.distance(feature_r[i], feature_r[i], False)
            Mxy = self.distance(feature_r[i], feature_f[i], False)
            Myy = self.distance(feature_f[i], feature_f[i], False)

            score[i * 7] = self.wasserstein(Mxy, True)
            score[i * 7 + 1] = self.mmd(Mxx, Mxy, Myy, 1)
            tmp = self.knn(Mxx, Mxy, Myy, 1, False)
            score[(i * 7 + 2):(i * 7 + 7)] = \
                tmp.acc, tmp.acc_real, tmp.acc_fake, tmp.precision, tmp.recall

        score[28] = self.inception_score(feature_f[3])
        score[29] = self.mode_score(feature_r[3], feature_f[3])
        score[30] = self.fid(feature_r[3], feature_f[3])

        return score


# metrics = pytorchMetrics()
#
#
# # Load images
# x_real = load_data('real/')
# x_fake = load_data('fake/')
# score = metrics.compute_score(x_real, x_fake)
# print(score)