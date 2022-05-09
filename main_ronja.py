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

def LRP_alex(img,alex):
    """
    Not working yet
    :param img:
    :param alex:
    :return:
    """
    img = np.array(cv2.imread('castle.jpg'))[..., ::-1] / 255.0
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
    img = img[np.newaxis]
    X = (torch.FloatTensor(np.transpose(img,(0, 3, 1, 2)) * 1) - mean) / std


    layers = list(alex._modules['features']) + utils.toconv(list(alex._modules['classifier']))
    L = len(layers)

    A = [X] +[None] * L
    for l in range(L): A[l + 1] = layers[l].forward(A[l])

    scores = np.array(A[-1].data.view(-1))
    ind = np.argsort(-scores)
    for i in ind[:10]:
        print('%20s (%3d): %6.3f' % (utils.imgclasses[i][:20], i, scores[i]))

    T = torch.FloatTensor((1.0 * (np.arange(1000) == 483).reshape([1, 1000, 1, 1])))

    R = [None] * L + [(A[-1] * T).data]

    for l in range(1, L)[::-1]:

        A[l] = (A[l].data).requires_grad_(True)

        if isinstance(layers[l], torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)

        if isinstance(layers[l], torch.nn.Conv2d) or isinstance(layers[l], torch.nn.AvgPool2d):

            if l <= 16:       rho = lambda p: p + 0.25 * p.clamp(min=0); incr = lambda z: z + 1e-9
            if 17 <= l <= 30: rho = lambda p: p;                       incr = lambda z: z + 1e-9 + 0.25 * (
                        (z ** 2).mean() ** .5).data
            if l >= 31:       rho = lambda p: p;                       incr = lambda z: z + 1e-9

            z = incr(utils.newlayer(layers[l], rho).forward(A[l]))  # step 1
            s = (R[l + 1] / z).data  # step 2
            (z * s).sum().backward();
            c = A[l].grad  # step 3
            R[l] = (A[l] * c).data  # step 4

        else:

            R[l] = R[l + 1]

    for i, l in enumerate([31, 21, 11, 1]):
        utils.heatmap(np.array(R[l][0]).sum(axis=0), 0.5 * i + 1.5, 0.5 * i + 1.5)

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


def LRP_vgg(img,vgg):

    img = np.array(cv2.imread('castle.jpg'))[..., ::-1] / 255.0
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
    img = img[np.newaxis]
    X = (torch.FloatTensor(np.transpose(img,(0, 3, 1, 2)) * 1) - mean) / std


    layers = list(vgg._modules['features']) + utils.toconv(list(vgg._modules['classifier']))
    L = len(layers)

    A = [X] +[None] * L
    for l in range(L): A[l + 1] = layers[l].forward(A[l])

    scores = np.array(A[-1].data.view(-1))
    ind = np.argsort(-scores)
    for i in ind[:10]:
        print('%20s (%3d): %6.3f' % (utils.imgclasses[i][:20], i, scores[i]))

    T = torch.FloatTensor((1.0 * (np.arange(1000) == 483).reshape([1, 1000, 1, 1])))

    R = [None] * L + [(A[-1] * T).data]

    for l in range(1, L)[::-1]:

        A[l] = (A[l].data).requires_grad_(True)

        if isinstance(layers[l], torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)

        if isinstance(layers[l], torch.nn.Conv2d) or isinstance(layers[l], torch.nn.AvgPool2d):

            if l <= 16:       rho = lambda p: p + 0.25 * p.clamp(min=0); incr = lambda z: z + 1e-9
            if 17 <= l <= 30: rho = lambda p: p;                       incr = lambda z: z + 1e-9 + 0.25 * (
                        (z ** 2).mean() ** .5).data
            if l >= 31:       rho = lambda p: p;                       incr = lambda z: z + 1e-9

            z = incr(utils.newlayer(layers[l], rho).forward(A[l]))  # step 1
            s = (R[l + 1] / z).data  # step 2
            (z * s).sum().backward();
            c = A[l].grad  # step 3
            R[l] = (A[l] * c).data  # step 4

        else:

            R[l] = R[l + 1]

    for i, l in enumerate([31, 21, 11, 1]):
        utils.heatmap(np.array(R[l][0]).sum(axis=0), 0.5 * i + 1.5, 0.5 * i + 1.5)

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

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    plt.imshow(npimg)
    plt.imsave("plot/example_cifar.svg",npimg)
    plt.show()

def main():
    img_size = 32
    transform = transforms.Compose(
        [transforms.Resize((img_size,img_size)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    #GTSRB_trainset = datasets.GTSRB(root='./data', split='train', download=True,transform=transform)

    #cifar_testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=None)

    loader = torch.utils.data.DataLoader(cifar_trainset,batch_size=4,shuffle=True)
    # get some random training images
    dataiter = iter(loader)
    images, labels = dataiter.next()
    print(images.shape)
    # show images
    imshow(torchvision.utils.make_grid(images))
    vgg = models.vgg16(pretrained=True)
    #print(vgg.eval())
    alex = models.alexnet(pretrained=True)
    #print(vgg.classifier.state_dict().keys())
    #print(alex.classifier.state_dict().keys())
    print(alex.eval())
    LRP_vgg(images,vgg)
if __name__ == '__main__':
    main()