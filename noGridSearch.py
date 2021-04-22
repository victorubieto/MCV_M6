import cv2
import numpy as np
from skimage.feature import match_template
from tqdm import trange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio


def evaluation_metrics(meth,target,ref):
    if meth != 'euclidean':
        method = eval(meth)
        res = cv2.matchTemplate(target, ref, method)

    return res

def metric_3d_plot(X, Y, Z, x_label, y_label, z_label):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

# 3.1 MSEN & PEPN
def read_flow(path_to_img):
    img = cv2.cvtColor(cv2.imread(path_to_img, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype(np.double)

    flow_u = (img[:, :, 0] - 2**15)/64
    flow_v = (img[:, :, 1] - 2**15)/64

    #_,flow_valid = cv2.threshold(img[:,:,2], 1, 1, cv2.THRESH_BINARY)
    flow_valid = img[:,:,2]
    flow_valid[flow_valid>1] = 1

    flow_u[flow_valid == 0] = 0
    flow_v[flow_valid == 0] = 0

    flow_img = np.dstack((flow_u, flow_v, flow_valid))
    #print(flow_img.shape)
    return flow_img

def evaluate_flow(flow_noc, flow):
    err = np.sqrt(np.sum((flow_noc[..., :2] - flow) ** 2, axis=2))
    noc = flow_noc[..., 2].astype(bool)
    msen = np.mean(err[noc] ** 2)
    pepn = np.sum(err[noc] > 3) / err[noc].size
    return msen, pepn

def msen_pepn(predicted_flow, gt_flow, motion_error_threshold=3):

    # MSEN
    u_difference = gt_flow[:, :, 0] - predicted_flow[:, :, 0]
    v_difference = gt_flow[:, :, 1] - predicted_flow[:, :, 1]
    squared_error = np.sqrt(u_difference ** 2 + v_difference ** 2)
    squared_error_non_occluded = squared_error[gt_flow[:, :, 2] != 0]

    msen = np.mean(squared_error_non_occluded)

    # PEPN
    n_wrong_pixels = np.sum(squared_error_non_occluded > motion_error_threshold)
    n_pixels_non_occ = len(squared_error_non_occluded)

    pepn = (n_wrong_pixels / n_pixels_non_occ) * 100

    return squared_error, msen, pepn

def calculate_adaptiveRegion(centerBlockj,w,searchArea,boolAdapt):
    if boolAdapt:
        if centerBlockj < w / 2:
            init_j = max(centerBlockj - searchArea / 2, 0)
            offset = min(centerBlockj - searchArea / 2, 0)
            end_j = centerBlockj + searchArea / 2 - offset
        elif centerBlockj == w / 2:
            init_j = centerBlockj - searchArea / 2
            end_j = centerBlockj + searchArea / 2
        elif centerBlockj > w / 2:
            end_j = min(centerBlockj + searchArea / 2, w)
            offset = max(centerBlockj + searchArea / 2, w) - w
            init_j = centerBlockj - searchArea / 2 - offset
    else:
        init_j = max(centerBlockj - searchArea / 2, 0)
        end_j = min(centerBlockj+searchArea/2,w)

    return init_j, end_j


def compute_block_matching(im1,im2,motion,searchArea,blockSize,quantStep):
    if motion == 'forward':
        refIm = im1
        targetIm = im2
    elif motion == 'backward':
        refIm = im2
        targetIm = im1

    h,w = np.shape(refIm)
    predicted_flow = np.zeros((h,w,3))

    method = 'cv2.TM_CCORR_NORMED'

    for i in trange(0,h-blockSize,blockSize):
        centerBlocki = int(i + blockSize / 2)
        init_i, end_i = calculate_adaptiveRegion(centerBlocki, h, searchArea, False)

        for j in range(0,w-blockSize,blockSize):
            centerBlockj = int(j + blockSize / 2)
            init_j ,end_j = calculate_adaptiveRegion(centerBlockj,w,searchArea,False)

            r = refIm[i:i+blockSize,j:j+blockSize]
            t = targetIm[int(init_i):int(end_i),int(init_j):int(end_j)]
            result = evaluation_metrics(method,t,r)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val<1.0:
                ci = cj = int(searchArea/2 - (blockSize / 2))
                if centerBlocki - searchArea/2 < 0:
                    ci = ci + (centerBlocki - searchArea/2)
                if centerBlockj - searchArea/2 < 0:
                    cj = cj + (centerBlockj - searchArea/2)

                if motion == 'forward':  # distance from the highest response to the center of the search space
                    flowVect = np.array(np.array(max_loc) - [cj, ci])
                else:
                    flowVect = np.array([cj, ci]) - np.array(max_loc)
            else:
                flowVect = [0, 0]

            predicted_flow[i:i+blockSize,j:j+blockSize] = np.array([flowVect[0], flowVect[1],1])


    return predicted_flow

def plotArrowsOP(flow_img, step, img):
    flow_img = cv2.resize(flow_img, (0, 0), fx=1. / step, fy=1. / step)
    u = flow_img[:, :, 0]
    v = flow_img[:, :, 1]
    x = np.arange(0, np.shape(flow_img)[0] * step, step)
    y = np.arange(0, np.shape(flow_img)[1] * step, step)
    U, V = np.meshgrid(y, x)
    M = np.hypot(u, v)
    plt.quiver(U, V, u, -v, M, cmap='Pastel2')
    plt.imshow(img, alpha=0.5, cmap='gray')
    plt.title('Orientation OF')
    plt.xticks([])
    plt.yticks([])
    plt.show()



def task11():
    # Read gt file
    im1 = cv2.imread('/home/mar/Desktop/M6/KITTI/data_stereo_flow/training/image_0/000045_10.png',
                          cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread('/home/mar/Desktop/M6/KITTI/data_stereo_flow/training/image_0/000045_11.png',
                          cv2.IMREAD_GRAYSCALE)
    flow_gt = read_flow('/home/mar/Desktop/M6/KITTI/data_stereo_flow/training/flow_noc/000045_10.png')

    motion = ['backward']
    blockSize = [16]
    searchAreas = [96]
    for m in motion:
        for i,bs in enumerate(blockSize):
            quantStep = bs
            for j,sa in enumerate(searchAreas):
                predicted_flow = compute_block_matching(im1, im2, m, sa, bs,quantStep)
                squared_error, msen, pepn = msen_pepn(predicted_flow,flow_gt)

                print('Motion: ',m)
                print('BS: ', bs)
                print('SA: ', sa)
                print('msen: ',msen)
                print('pepn: ',pepn)

def task21():
    cap = cv2.VideoCapture("beer_non-stabilized.mp4")
    seq_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Reference image
    previous_frame = cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
    w = previous_frame.shape[0]
    h = previous_frame.shape[1]

    # stabilized video sequence
    stabilized_sequence = []
    stabilized_sequence.append(previous_frame)

    # cummulative deviation respect to initial frame
    deviation_x = 0.0
    deviation_y = 0.0
    for i in range(seq_length-1):

        # Current frame
        current_frame = cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)

        # predict flow with block matching
        blockSize = 16
        searchArea = 96
        quantStep = 16
        predicted_flow = compute_block_matching(previous_frame[:,:,0], current_frame[:,:,0], 'backward', searchArea, blockSize, quantStep)

        # deviation as median of flow vectors. mean was affected by actual movement of objects
        vector_u = predicted_flow[:, :, 1]
        vector_v = predicted_flow[:, :, 0]
        deviation_x += np.median(vector_u)
        deviation_y += np.median(vector_v)

        # stabilizing frame using homography
        H = np.array([[1, 0, -deviation_y], [0, 1, -deviation_x]], dtype=np.float32)
        stabilized_frame = cv2.warpAffine(current_frame, H, (h, w)) 
        stabilized_sequence.append(stabilized_frame)

        # update previous frame as new reference
        previous_frame = current_frame

    kargs = { 'duration': fps }
    imageio.mimsave('median_beer_stabilized.gif', stabilized_sequence,format='GIF', fps=fps)

    #squared_error, msen, pepn = msen_pepn(predicted_flow,flow_gt)

    #plt.imshow(predicted_flow)
    #plt.show()


if __name__ == '__main__':

    #task11()
    #task12_B()
    #task12_C()
    task21()
    #task22()



