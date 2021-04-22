import pickle
import numpy as np
import os
import cv2
from metric_learn import NCA
import random

from tqdm import trange
from histograms import *


def format_pkl(all_pkl,camerasL):
    allDetections = []
    for i,cam in enumerate(camerasL):
        data = []
        for j,id in enumerate(all_pkl[i]['id']):
            detections = []
            list_frames = all_pkl[i]['frame'][j]
            if len(np.where(np.array(list_frames)==-1)[0])>0:
                del list_frames[-1]
            for k,frame in enumerate(list_frames):
                boxes = all_pkl[i]['box'][j][k]
                detections.append({'frame':frame,'box':boxes})
            data.append({'track_id':id,'info':detections})

        allDetections.append({cam:data})

    return allDetections

def format_gt_pkl(all_pkl,camerasL):

    allDetections = []
    for i,cam in enumerate(camerasL):
        data = []
        for j,id in enumerate(all_pkl[i]['id']):
            detections = []
            list_frames = all_pkl[i]['frame'][j]
            if len(np.where(np.array(list_frames)==-1)[0])>0:
                del list_frames[-1]
            for k,frame in enumerate(list_frames):
                boxes = all_pkl[i]['box'][j][k]
                detections.append({'frame':frame,'box':boxes})
            data.append({'track_id':id,'info':detections})

        allDetections.append({cam:data})

    return allDetections

def import_gt_track(gt_path, outpath='gt_tracks.pkl', save=False):

    dict = {}
    ids = []
    frames = {}
    bboxes = {}

    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split(",")

            id = int(fields[1])
            frame = int(fields[0])
            bbox = []
            bbox.append(int(fields[2]))
            bbox.append(int(fields[3]))
            bbox.append(int(fields[4]))
            bbox.append(int(fields[5]))

            if id in ids:
                frames[id].append(frame)
                bboxes[id].append(bbox)
            else:
                ids.append(id)
                frames[id] = [frame]
                bboxes[id] = [bbox]

    dict['id'] = ids
    dict['frame'] = list(frames.values())
    dict['box'] = list(bboxes.values())

    if save:
        outfile = open(outpath, 'wb')
        pickle.dump(dict, outfile)
        outfile.close()

    return dict


def get_gt_info(gt_paths):

    camerasL = []
    data = {}
    data['id'] = []
    data['frame'] = []
    data['box'] = []
    data['cam'] = []
    vid_paths = {}

    for seq_path in gt_paths:
        for cam_path in os.listdir(seq_path):

            camerasL.append(cam_path)
            gt_file_path = os.path.join(seq_path,cam_path,'gt','gt.txt')
            vid_paths[cam_path] = os.path.join(seq_path,cam_path,'vdo.avi')
            gt_data = import_gt_track(gt_file_path)

            data['id'].extend(gt_data['id'])
            data['frame'].extend(gt_data['frame'])
            data['box'].extend(gt_data['box'])
            data['cam'].extend([cam_path] * len(gt_data['id']))

    id,fr,bo,ca = zip(*sorted(zip(data['id'],data['frame'],data['box'],data['cam'])))
    data['id'], data['frame'], data['box'], data['cam'] = list(id),list(fr),list(bo),list(ca)

    return data, vid_paths

def train_NCA(gtdata, vid_paths):

    # set number of images to get from each track
    num_track = 20

    if os.path.isfile('out_features.pkl'):
        infile = open('out_features.pkl', 'rb')
        feat_dict = pickle.load(infile)
        infile.close()

        new_features = feat_dict['features']
        new_labels = feat_dict['labels']

    else:
        new_features = []
        new_labels = []
        uniq_tracks = np.unique(gtdata['id'])
        for id_tr in trange(len(uniq_tracks), desc="get gt data"):
            track_id = uniq_tracks[id_tr]

            # Get all the indices that have the same tracking number
            indices = [i for i, x in enumerate(gtdata['id']) if x == track_id]

            frames = gtdata['frame']
            frames = [frames[i] for i in indices]
            bboxes = gtdata['box']
            bboxes = [bboxes[i] for i in indices]
            cameras = gtdata['cam']
            cameras = [cameras[i] for i in indices]

            cam_frames = []
            for i in range(len(frames)):
                cam_frames.append([cameras[i]] * len(frames[i]))

            #flatten lists
            frames = [item for sublist in frames for item in sublist]
            bboxes = [item for sublist in bboxes for item in sublist]
            cam_frames = [item for sublist in cam_frames for item in sublist]


            if len(frames) > num_track:
                indices = random.sample(range(len(frames)), num_track)
                frames = [frames[i] for i in indices]
                bboxes = [bboxes[i] for i in indices]
                cam_frames = [cam_frames[i] for i in indices]

            for i in range(len(frames)):

                # Get bbox from image
                vidpath = vid_paths[cam_frames[i]]
                cap = cv2.VideoCapture(vidpath)
                total_frames = cap.get(7)
                cap.set(1, frames[i]-1)
                ret, vid_frame = cap.read()
                bb = bboxes[i]
                bbox_img = cv2.cvtColor(vid_frame[bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2],:], cv2.COLOR_BGR2RGB)

                box_features = histogram_multires(bbox_img)
                # box_features.extend(rgbHist([bbox_img], 128)[0])
                # box_features.extend(hsvHist([bbox_img], 128)[0])
                # box_features.extend(labHist([bbox_img], 128)[0])

                new_features.append(box_features)
                new_labels.append(track_id)
                cap.release()

        filename = 'out_features.pkl'
        outfile = open(filename, 'wb')
        pickle.dump({'features': new_features, 'labels': new_labels}, outfile)
        outfile.close()

    X = np.array(new_features)
    Y = np.array(new_labels)
    nca = NCA(init='pca', n_components=400, max_iter=1500, verbose=True)
    nca.fit(X, Y)

    filename = 'multires.pkl'
    outfile = open(filename, 'wb')
    pickle.dump(nca,outfile)
    outfile.close()

    return nca

if __name__ == '__main__':

    gt_paths = [r"C:\Users\Alex\Downloads\aic19-track1-mtmc-train\train\S01",
               r"C:\Users\Alex\Downloads\aic19-track1-mtmc-train\train\S04"]

    gtdata, vid_paths = get_gt_info(gt_paths)
    nca_classif = train_NCA(gtdata, vid_paths)
