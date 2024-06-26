from errno import WSAEDQUOT
import numpy as np


class Adam:
    def __init__(self, lr=0.1, beta1=0.9, beta2=0.99, weight_decay=0.1,
                 epsilon=1e-8, AdamW=False):  # weight_decay: for AdamW
    
        print('The optimizer is Adam.')
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.AdamW = AdamW
        self.mt = 0
        self.vt = 0
        self.t = 0

    def run(self, params, grad):
        self.t += 1
        self.mt = self.beta1 * self.mt + (1-self.beta1) * grad
        self.vt = self.beta2 * self.vt + (1-self.beta2) * (grad * grad)
        if self.AdamW:
            m_hat = self.mt / (1-self.beta1**self.t)
            v_hat = self.vt / (1-self.beta2**self.t)
            params -= self.lr * (m_hat / (v_hat**0.5 + self.epsilon)+self.weight_decay*params)
        else:
            m_hat = self.mt / (1-self.beta1**self.t)
            v_hat = self.vt / (1-self.beta2**self.t)
            params -= self.lr * m_hat / (self.epsilon + np.sqrt(v_hat))
        return params

class RMSProp:
    def __init__(self, lr=0.005, beta=0.999, e=1e-8):
        self.e = e
        self.r = 0
        self.lr = lr
        self.beta = beta
    def run(self, params, grad):
        self.r = self.r * self.beta + (1-self.beta)*np.square(grad)
        return np.array(params) - self.lr * grad / np.sqrt(self.r + self.e)
    
class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr
    def run(self, params, grad):
        return np.array(params)-self.lr*np.array(grad)