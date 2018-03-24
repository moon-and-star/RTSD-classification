#!/usr/bin/env python
import cv2
import numpy as np
from shutil import copyfile, rmtree
from util.util import safe_mkdir


def histeq(img):
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    ycbcr[:,:,0] = cv2.equalizeHist(ycbcr[:,:,0])
    return cv2.cvtColor(ycbcr, cv2.COLOR_YCR_CB2BGR)



def autoContrast(img, frac=0.1):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    Y = lab[:,:, 0].astype(np.float32)
    Ysorted = np.sort(Y.reshape(-1))
    m = int(len(Ysorted) * frac)
    Ysorted = Ysorted[m:-m]

    ymin, ymax = Ysorted[0], Ysorted[-1]
    alpha = 255. / (ymax-ymin)
    # print((Y - ymin) * alpha)
    # print ((ymax - ymin) * alpha )

    Y =  ((Y - ymin) * alpha)
    Y[Y < 0] = 0
    Y[Y > 255] = 255
    lab[:,: ,0] = Y.astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)



def adaHE(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2,2))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def getPad(origin, radius, shape):
    pad = 0
    if origin[0] - radius < 0:
        pad = radius - origin[0]
    if origin[1] - radius < 0:
        pad = max(pad, radius - origin[1])
    if int(origin[0] + radius) >= shape[1]:
        pad = max(pad, origin[0] + radius - shape[1])
    if int(origin[1] + radius) >= shape[0]:
        pad = max(pad, origin[1] + radius - shape[0])
    pad = int(pad) + 3

    return pad


def getResceled(img, x_scale, y_scale):
    interpolation = cv2.INTER_LANCZOS4
    if x_scale < 0.5 or y_scale < 0.5:
        interpolation = cv2.INTER_AREA
    return cv2.resize(img, (0, 0), fx = x_scale, fy = y_scale, interpolation = interpolation)
 

def getBordered(img, pad, border_type):
    if border_type == 'replicate':
        res = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    elif border_type == 'black':
        res = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT,value=[0,0,0])
    elif border_type == 'grey':
        res = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT,value=[128,128,128])
    else:
        print('ERROR: unknown border type!')
        exit()
    return res


def crop(img, x1, y1, x2, y2, expand = 0, size=48, border='replicate'):
    dx, dy = x2 - x1, y2 - y1
    x_scale, y_scale = float(size) / dx, float(size) / dy
    rescaled = getResceled(img, x_scale, y_scale)

    origin = x_scale * (x1 + x2) / 2.0, y_scale * (y1 + y2) / 2.0
    radius = int(size / 2 + expand)
    pad = getPad(origin, radius, rescaled.shape)
    
    bordered = getBordered(rescaled, pad, border)
    origin = list(map(int, origin))
    return bordered[pad + origin[1] - radius : pad + origin[1] + radius, pad + origin[0] - radius : pad + origin[0] + radius]



def accumulate(mean, transformed):
    mean[0] += transformed[:, :, 0]
    mean[1] += transformed[:, :, 1]
    mean[2] += transformed[:, :, 2]



def saveImgInfo(outpath, phase, mean, total, new_gt, mapping):
    #caffe works with openCV, so the order of channels is BGR
    with open("{}/{}/mean.txt".format(outpath, phase), "w") as f:
        b, g, r = np.mean(mean[0]), np.mean(mean[1]), np.mean(mean[2]) 
        f.write("{} {} {}".format(b, g, r))

        if max(b, g, r) < 50:
            print('WARNING: low mean values\nb={}, g={}, r={}'.format(b,g,r))


    with open("{}/{}_size.txt".format(outpath, phase), 'w') as f:
        f.write(str(total))

    with open("{}/gt_{}_full.txt".format(outpath, phase), 'w') as f: #create file to write labels in
        for i in new_gt:
            f.write(i + '\n')

    with open("{}/mapping_{}.txt".format(outpath, phase), 'w') as f:
        for i in sorted(mapping):
            f.write("{:15} {:15}\n".format(i, mapping[i]))

    src = "{}/gt_{}_full.txt".format(outpath, phase)
    dst = "{}/gt_{}.txt".format(outpath, phase)
    copyfile(src, dst)




rate = 100

def process(rootpath, outpath, phase, size=32, pad=4, border='replicate'):
    #preparations
    with  open('{}/gt_{}.txt'.format(rootpath, phase)) as f: # open file to read labels (and, may be coords)
        markup = f.readlines() 
    mean = np.zeros((3, size + pad * 2, size + pad * 2), dtype=np.float32)
    total = 0; new_gt = []; mapping = {}

    for image_name, clid in sorted([x.replace('\r\n', '').split(',') for x in markup]):
        clid = int(clid)
        if total % rate == 0:
            print(image_name)

        #read and trasform
        img = cv2.imread("{}/{}/{}".format(rootpath,phase,image_name))
        m, n, _ = img.shape
        transformed = crop(img, x1=0, y1=0, y2=m-1, x2=n-1, expand=pad,size=size, border=border)
        # transformed = histeq(transformed)
        accumulate(mean, transformed)

        #save transformed
        safe_mkdir("{}/{}/{}".format(outpath, phase, clid))
        cv2.imwrite("{}/{}/{}/{}.png".format(outpath, phase, clid, str(total).zfill(6)), transformed)

        #update
        new_name = "{}/{}.png".format(clid, str(total).zfill(6))
        mapping[new_name] = image_name
        new_gt += ["{} {}".format(new_name, clid)]
        total = total + 1


    mean = mean / float(total)
    saveImgInfo(outpath, phase, mean, total, new_gt, mapping)
    
