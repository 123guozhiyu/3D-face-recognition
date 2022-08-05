# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import open3d as o3d
import math
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from PIL import Image
import struct
# rescale(Phi,0,255)
def rescale(x, a, b):

    m = np.nanmin(x[:])
    M = np.nanmax(x[:])
    y = (b - a) * (x - m)/(M - m) + a
    return y

def read_obj(root):
    with open(root) as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "f":
                break
    # points转变为矩阵，方便处理
    points = np.array(points)
    return points

def cart2sph(x, y, z):
    xy = np.sqrt(x ** 2 + y ** 2)  # sqrt(x² + y²)
    x_2 = x ** 2
    y_2 = y ** 2
    z_2 = z ** 2
    r = np.sqrt(x_2 + y_2 + z_2)  # r = sqrt(x² + y² + z²)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, xy)
    return r, theta, phi

def get_normals(points_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_cloud)

    # 通过点云获取点云的法向量
    # o3d.geometry.estimate_normals(
    #     pcd,
    #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
    #                                                       max_nn=30))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    return pcd

def img_norm(img):
    """
    把图片中的像素值映射到[0~255]
    """
    img_max = (img[img != 0]).max()
    img_min = (img[img != 0]).min()
    img_new = (img-img_min)*255.0/(img_max-img_min)
    th = (0-img_min)*255.0/(img_max - img_min)
    img_new[img_new==th] = 0
    # 用用中值模糊去掉噪点
    img_new = cv2.medianBlur(img_new.astype(np.float32), 5)
    return img_new

def find_normals(im, tri):

    nface = tri.shape[0]
    nvert = im.shape[0]
    normal = np.zeros((nvert, 3), dtype=np.float32)

    normalf = np.cross(im[tri[:, 1], :] - im[tri[:, 0], :],
                       im[tri[:, 2], :] - im[tri[:, 0], :])

    d = np.sqrt(np.sum(normalf**2, 1))
    d[d < np.spacing(1)] = 1
    d = d[:, np.newaxis]
    normalf = normalf / d

    for i in  range(nface):
        f = tri[i,:]
        for j in range(3):
            normal[f[j], :] = normal[f[j], :] + normalf[i, :]

    d = np.sqrt(np.sum(normal**2, axis=1))
    d[d < np.spacing(1)] = 1
    d = d[:, np.newaxis]
    normal = normal / d
    me = np.mean(im, axis=1)
    me = me[:, np.newaxis]
    v = im - me
    s = np.sum(v*normal, axis=0)
    if np.sum(s > 0) < np.sum(s < 0):
        normal = -normal
        normalf = -normalf
    return normal

def point2img(pcd):
    points = np.array(pcd.points)
    x, y, z = points[:,0].tolist(), points[:,1].tolist(), points[:,2].tolist()

    normals = np.array(pcd.normals)
    alpha, beta, theta = normals[:,0].tolist(), normals[:,1].tolist(), normals[:,2].tolist()

    u_list, v_list, z_list = [], [], []
    for i, j, k in zip(x, y, z):
        u_list.append((i*616.009)/k)
        v_list.append((j*614.024)/k)
        z_list.append(k)
    width = int(max(u_list) - min(u_list))
    height = int(max(v_list) - min(v_list))
    gray_img = np.zeros((width+1, height+1, 1))
    alpha_img = np.zeros((width+1, height+1, 1))
    theta_img = np.zeros((width+1, height+1, 1))
    img_3d = np.zeros((width+1, height+1, 3))
    u_min = min(u_list)
    v_min = min(v_list)
    u_list = [int(i-u_min) for i in u_list]
    v_list = [int(i-v_min) for i in v_list]

    for u, v, z, al, th in zip(u_list, v_list, z_list, alpha, theta):
        gray_img[u,v] = z
        alpha_img[u,v] = math.acos(abs(al))
        theta_img[u,v] = math.acos(abs(th))

    img_gray = img_norm(gray_img)
    alpha_img = img_norm(alpha_img)
    theta_img = img_norm(theta_img)

    img_3d[:,:,0] = img_gray
    img_3d[:,:,1] = alpha_img
    img_3d[:,:,2] = theta_img

    cv2.imwrite("result/depth.jpg", img_gray)
    cv2.imwrite("result/alpha.jpg", alpha_img)
    cv2.imwrite("result/theta.jpg", theta_img)

    ## 左右对称补洞
    # for u in range(width-1):
    #     for v in range(height-1):
    #         if img_3d[u,v,2] == 0 and img_3d[u,v,1] == 0 and img_3d[u,v,0] == 0 and width*0.45 < 2*u_nose-u < width*0.55:
    #             img_3d[u,v,:] = img_3d[2*u_nose-u, v, :]
    return img_3d

def readbc(file):
    npoints = os.path.getsize(file) // 4
    with open(file,'rb') as f:
        raw_data = struct.unpack('f'*npoints, f.read(npoints*4))
        data = np.asarray(raw_data,dtype=np.float32)
#    data = data.reshape(len(data)//6, 6)
    data = data.reshape(3, len(data)//3)
    # translate the nose tip to [0,0,0]
#    data = (data[:,0:2] - data[8157,0:2]) / 100
    return data.T

if __name__ == '__main__':

    name_list = os.listdir('data/001')
    for i in name_list:
        points = readbc('data/001/{}'.format(i))
        nos_ind = np.argmax(points[:, 2])
        NostTip_Z = points[nos_ind, 2]
        NostTip_Y = points[nos_ind, 1]
        NostTip_X = points[nos_ind, 0]
        NostTip = [NostTip_X, NostTip_Y, NostTip_Z]
        NostTip = np.array(NostTip)
        idx = np.sqrt(np.sum((points - NostTip)**2, axis=1))
        points = points[idx < 90, :]
        tri = Delaunay(points[:, 0:2])
        tri = tri.simplices
        xmin = min(points[:, 0])
        ymin = min(points[:, 1])
        xmax = max(points[:, 0])
        ymax = max(points[:, 1])
        scale = 0.4
        X1, Y1 = np.meshgrid(np.arange(xmin, xmax, scale), np.arange(ymax, ymin, -scale))
        X2 = np.arange(xmin, xmax, scale)
        Y2 = np.arange(ymin, ymax, scale)
        Zd = griddata(points[:, 0:2], points[:, 2], (X1, Y1))

        pcd = get_normals(points)
        normals = np.array(pcd.normals)
        norm = find_normals(points, tri)

        _, theta, phi = cart2sph(norm[:, 0], norm[:, 1], norm[:, 2])
        if np.sum(phi) < 0:
            phi = -phi

        phi = griddata(points[:, 0:2], phi, (X1, Y1))
        theta = griddata(points[:, 0:2], np.abs(theta), (X1, Y1))

        Zd = (rescale(Zd, 0, 255)).astype(np.uint8)
        phi = (rescale(phi, 0, 255)).astype(np.uint8)
        theta = (rescale(theta, 0, 255)).astype(np.uint8)
        phi[Zd==0] = 0
        theta[Zd==0]=0
        sh = phi.shape
        I = np.zeros((sh[0], sh[1], 3))
        I[:, :, 0] = Zd
        I[:, :, 1] = phi
        I[:, :, 2] = theta
        I = I.astype(np.uint8)
        H = sh[0]
        W = sh[1]
        if H >= W:
            I = cv2.resize(I, (int(np.ceil((512/H)*W)), 512))
        else:
            I = cv2.resize(I, (512, int(np.ceil((512/H)*W))))

        new_sh = I.shape[0:2]
        temp = min(new_sh)
        up = int(np.floor((512-temp)/2))
        down = int(np.ceil((512-temp)/2))
        if H > W:
            I = np.pad(I, ((0, 0), (up, down), (0, 0)), 'constant')
        else:
            I = np.pad(I, ((up, down), (0, 0), (0, 0)), 'constant')

        I = cv2.resize(I, (256, 256))
        I1 = I[:,:,::-1]
        cv2.imwrite('result/{}.jpg'.format(i.split('.')[0]),I1)
    # img_3d = point2img(pcd)




