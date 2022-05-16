import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt
import numpy as np
import utils
import scipy
import cv2

import torchshape


def LRP(picture,model,model_str,save):
    img = np.array(cv2.imread(picture))

    img = np.asarray(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC))
    img = (img[..., ::-1] / 255.0)

    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
    X = (torch.FloatTensor(img[np.newaxis].transpose([0, 3, 1, 2]) * 1) - mean) / std


    layers = list(model._modules['features']) + utils.toconv(list(model._modules['classifier']),model_str)
    L = len(layers)

    #TODO nograd ?
    A = [X] +[None] * L
    for l in range(L):
        A[l + 1] = layers[l].forward(A[l])
    print(A[-1].shape)
    scores = np.array(A[-1].data.view(-1))
    ind = np.argsort(-scores)
    for i in ind[:10]:
        print('%20s (%3d): %6.3f' % (utils.imgclasses[i][:20], i, scores[i]))

    topClass = ind[0]
    T = torch.FloatTensor((1.0 * (np.arange(1000) == topClass).reshape([1, 1000, 1, 1]))) #mask of output class
    R = [None] * L + [(A[-1] * T).data] # Relevance list, with las being T
    for l in range(1, L)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)

        if isinstance(layers[l], torch.nn.MaxPool2d):layers[l] = torch.nn.AvgPool2d(2)

        if isinstance(layers[l], torch.nn.Conv2d) or isinstance(layers[l], torch.nn.AvgPool2d):
            # roh rules
            if l <= 16:       rho = lambda p: p + 0.25 * p.clamp(min=0); incr = lambda z: z + 1e-9
            if 17 <= l <= 30: rho = lambda p: p;                       incr = lambda z: z + 1e-9 + 0.25 * (
                        (z ** 2).mean() ** .5).data
            if l >= 31:       rho = lambda p: p;                       incr = lambda z: z + 1e-9

            # lRP math
            z = incr(utils.newlayer(layers[l], rho).forward(A[l]))  # step 1

            s = (R[l + 1] / z).data  # step 2
            (z * s).sum().backward();
            c = A[l].grad  # step 3
            R[l] = (A[l] * c).data  # step 4

        else:

            R[l] = R[l + 1]
    if model_str == "alex":
        layers_map = [15,10,7,1]
    else :
        layers_map = [31, 21, 11, 1]

    name = picture.rsplit("/")[-1]
    name = name.rsplit(".")[0]
    name = name +"_" +model_str
    for i, l in enumerate(layers_map):
        if i == 1: utils.heatmap(np.array(R[l][0]).sum(axis=0), 0.5 * i + 1.5, 0.5 * i + 1.5,name,save)
        else:   utils.heatmap(np.array(R[l][0]).sum(axis=0), 0.5 * i + 1.5, 0.5 * i + 1.5)

    A[0] = (A[0].data).requires_grad_(True)

    lb = (A[0].data * 0 + (0 - mean) / std).requires_grad_(True)
    hb = (A[0].data * 0 + (1 - mean) / std).requires_grad_(True)

    z = layers[0].forward(A[0]) + 1e-9  # step 1 (a)
    z -= utils.newlayer(layers[0], lambda p: p.clamp(min=0)).forward(lb)  # step 1 (b)
    z -= utils.newlayer(layers[0], lambda p: p.clamp(max=0)).forward(hb)  # step 1 (c)
    s = (R[1] / z).data  # step 2
    (z * s).sum().backward()
    c, cp, cm = A[0].grad, lb.grad, hb.grad  # step 3
    R[0] = (A[0] * c + lb * cp + hb * cm).data

    utils.heatmap(np.array(R[0][0]).sum(axis=0), 3.5, 3.5)

    pass

def imshow(img, save=False):
    img = img / 2 + 0.5     # unnormalize
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    plt.imshow(npimg)
    if save: plt.imsave("plot/example_cifar.svg",npimg)
    plt.show()

def main():

    vgg = models.vgg16(pretrained=True)
    alex = models.alexnet(pretrained=True)
    LRP('pictures/ILSVRC2012_val_00000001.JPEG',vgg,model_str="vgg",save=True)

if __name__ == '__main__':
    main()