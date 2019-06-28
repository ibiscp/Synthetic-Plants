from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import math
from tensorflow.keras.models import Model
import tensorflow
from tensorflow.keras.layers import Input
import cv2
import ot
from scipy import linalg
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Score:
    emd = None
    mmd = None
    knn = None
    inception = None
    mode = None
    fid = None

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

class Metrics(object):
    def __init__(self):
        # Model
        self.model = InceptionV3(weights='imagenet', include_top=True)

        # Layer names
        feature_name = 'mixed10'
        output_name = 'avg_pool'

        self.features_model = Model(inputs=self.model.input, outputs=self.model.get_layer(feature_name).output)
        self.logits_model = Model(inputs=self.model.input, outputs=self.model.get_layer(output_name).output)

    def calculate_features(self, images):
        features = self.features_model.predict(images)
        logits = self.logits_model.predict(images)
        softmax = self.model.predict(images)
        return images, features, logits, softmax

    def distance(self, X, Y, sqrt):
        nX = X.shape[0]
        nY = Y.shape[0]
        X = X.reshape((nX, -1))
        X2 = np.sum(X * X, axis=1).reshape((nX, 1))
        Y = Y.reshape((nY, -1))
        Y2 = np.sum(Y * Y, axis=1).reshape((nY, 1))

        M = np.repeat(X2, nY, axis=1) + np.transpose(np.repeat(Y2, nX, axis=1)) - 2 * np.dot(X, np.transpose(Y))

        del X, X2, Y, Y2

        if sqrt:
            M = np.sqrt((M + np.absolute(M)) / 2)

        return M

    def wasserstein(self, M, sqrt):
        if sqrt:
            M = np.sqrt(np.absolute(M))
        emd = ot.emd2([], [], M)

        return emd

    def mmd(self, Mxx, Mxy, Myy, sigma):
        scale = np.mean(Mxx)
        Mxx = np.exp(-Mxx / (scale * 2 * sigma * sigma))
        Mxy = np.exp(-Mxy / (scale * 2 * sigma * sigma))
        Myy = np.exp(-Myy / (scale * 2 * sigma * sigma))
        mmd = math.sqrt(Mxx.mean() + Myy.mean() - 2 * Mxy.mean())

        return mmd

    def knn(self, Mxx, Mxy, Myy, k, sqrt):
        n0 = Mxx.shape[0]
        n1 = Myy.shape[0]
        label = np.concatenate((np.ones(n0), np.zeros(n1)))
        M = np.concatenate((np.concatenate((Mxx, Mxy), 1), np.concatenate((np.transpose(Mxy), Myy), 1)), 0)
        if sqrt:
            M = np.sqrt(np.absolute(M))
        INFINITY = float('inf')
        temp = (M + np.diag(INFINITY * np.ones(n0 + n1)))
        idx = np.argsort(temp, axis=0)[0:k].reshape((k,-1))
        #val = np.choose(idx, temp.T)#temp[idx]

        count = np.zeros(n0 + n1)
        for i in range(0, k):
            count = count + np.choose(idx[i], label)
        pred = np.greater_equal(count, k/2*np.ones(n0 + n1)).astype(float)

        s = Score_knn()
        s.tp = np.sum(pred * label)
        s.fp = np.sum(pred * (1 - label))
        s.fn = np.sum((1 - pred) * label)
        s.tn = np.sum((1 - pred) * (1 - label))
        s.precision = s.tp / (s.tp + s.fp + 1e-10)
        s.recall = s.tp / (s.tp + s.fn + 1e-10)
        s.acc_real = s.tp / (s.tp + s.fn)
        s.acc_fake = s.tn / (s.tn + s.fp)
        s.acc = np.mean(np.equal(label, pred).astype(float))
        s.k = k

        return s

    def inception_score(self, X, eps=1e-20):
        kl = X * (np.log(X + eps) - np.repeat(np.log(np.mean(X, axis=0) + eps).reshape((1, -1)), X.shape[0], axis=0))
        score = np.exp(np.mean(np.sum(kl, axis=1)))

        return float(score)

    def mode_score(self, X, Y, eps=1e-20):
        kl1 = X * (np.log(X + eps) - np.repeat(np.log(np.mean(X, axis=0) + eps).reshape((1, -1)), X.shape[0], axis=0))
        kl2 = np.mean(X, axis=0) * (np.log(np.mean(X, axis=0) + eps) - np.log(np.mean(Y, axis=0) + eps))
        score = np.exp(np.mean(np.sum(kl1, axis=1)) - np.sum(kl2))

        return float(score)

    def fid(self, X, Y):
        m = np.mean(X, axis=0)
        m_w = np.mean(Y, axis=0)

        C = np.cov(X.transpose())
        C_w = np.cov(Y.transpose())
        C_C_w_sqrt = linalg.sqrtm(C.dot(C_w), True).real

        score = m.dot(m) + m_w.dot(m_w) - 2 * m_w.dot(m) + \
                np.trace(C + C_w - 2 * C_C_w_sqrt)
        return np.sqrt(score)

    def compute_score(self, real, fake, k=1, sigma=1, sqrt=True):

        # np.random.seed(0)
        # feature_r = [np.random.rand(5, 299, 299, 3), np.random.rand(5, 8, 8, 2048), np.random.rand(5, 2048), np.random.rand(5, 1000)]
        # feature_f = [np.random.rand(5, 299, 299, 3), np.random.rand(5, 8, 8, 2048), np.random.rand(5, 2048), np.random.rand(5, 1000)]

        start = time.time()

        # Features
        feature_r = self.calculate_features(real)
        feature_f = self.calculate_features(fake)

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

        print('Time', time.time() - start)

        # # 4 feature spaces and 7 scores + incep + modescore + fid
        # score = np.zeros(4 * 7 + 3)
        # for i in range(0, 4):
        #     print('compute score in space: ' + str(i))
        #     Mxx = self.distance(feature_r[i], feature_r[i], False)
        #     Mxy = self.distance(feature_r[i], feature_f[i], False)
        #     Myy = self.distance(feature_f[i], feature_f[i], False)
        #
        #     score[i * 7] = self.wasserstein(Mxy, True)
        #     score[i * 7 + 1] = self.mmd(Mxx, Mxy, Myy, 1)
        #     tmp = self.knn(Mxx, Mxy, Myy, 1, False)
        #     score[(i * 7 + 2):(i * 7 + 7)] = \
        #         tmp.acc, tmp.acc_real, tmp.acc_fake, tmp.precision, tmp.recall
        #
        # score[28] = self.inception_score(feature_f[3])
        # score[29] = self.mode_score(feature_r[3], feature_f[3])
        # score[30] = self.fid(feature_r[3], feature_f[3])

        return s

# images = load_data(repeat = true)
#
# ibis = Metrics()
# score = ibis.compute_score(images[0:5],images[5:10])