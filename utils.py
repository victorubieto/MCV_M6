import numpy as np
import cv2
import os
import motmetrics as mm
from tqdm import trange
import NCA_train

def euclid_dist(point1, point2):
    d = np.linalg.norm(np.array(point1) - np.array(point2))
    return d

def centroid(bboxGT):  # x, y, w, h
    x1 = bboxGT[0]
    y1 = bboxGT[1]
    x2 = bboxGT[2] + x1
    y2 = bboxGT[3] + y1
    xCenter = (x1 + x2) / 2
    yCenter = (y1 + y2) / 2
    return xCenter, yCenter

def iou(boxA, boxB):
    # For each prediction, compute its iou over all the boxes in that frame
    x11, y11, x12, y12 = np.split(boxA, 4, axis=0)
    x21, y21, x22, y22 = np.split(boxB, 4, axis=0)

    # Calculate the intersection in the bboxes
    xmin = np.maximum(x11, np.transpose(x21))
    ymin = np.maximum(y11, np.transpose(y21))
    xmax = np.minimum(x12, np.transpose(x22))
    ymax = np.minimum(y12, np.transpose(y22))
    w = np.maximum(xmax - xmin + 1.0, 0.0)
    h = np.maximum(ymax - ymin + 1.0, 0.0)
    intersection = w * h

    # Union
    areaboxA = (x12 - x11 + 1.0) * (y12 - y11 + 1.0)
    areaboxB = (x22 - x21 + 1.0) * (y22 - y21 + 1.0)
    union = areaboxA + np.transpose(areaboxB) - intersection

    iou = intersection / union

    return iou

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def show_NCA_pairs(name,cropped_bboxes_cam1, cropped_bboxes_cam2):

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (1,1)
    fontScale = 0.2
    fontColor = (255, 255, 255)
    lineType = 2

    imgs = []
    for i in range(len(cropped_bboxes_cam1)):
        imgs.append(hconcat_resize_min([np.array(cropped_bboxes_cam1[i]),np.array(cropped_bboxes_cam2[i])]))

    img = vconcat_resize_min(imgs)
    cv2.imwrite(name, img)

def reformat_predictions(correct_pred):
    boxes_List = []
    frames_List = []
    trackId_List = []
    cams_List = []
    for i in range(len(correct_pred)):
        cam = list(correct_pred[i].keys())[0]
        for j in range(len(correct_pred[i][cam])):
            track_id =correct_pred[i][cam][j]['track_id']
            for k in range(len(correct_pred[i][cam][j]['info'])):
                frame = correct_pred[i][cam][j]['info'][k]['frame']
                boxes = correct_pred[i][cam][j]['info'][k]['box']
                cams_List.append(cam)
                trackId_List.append(track_id)
                frames_List.append(frame)
                boxes_List.append(boxes)

    frames_List, boxes_List, trackId_List, cams_List = zip(*sorted(zip(frames_List, boxes_List, trackId_List, cams_List)))
    return list(cams_List),list(trackId_List),list(frames_List),list(boxes_List)

def crop_image(track,index_box_track):
    croppedIms = []
    for i in range(len(index_box_track)):
        id= index_box_track[i]
        bbox = track['info'][int(id* len(track['info']))]['box']
        bbox = [int(p) for p in bbox]
        path = track['info'][int(id* len(track['info']))]['frame_path']
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropIm = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        croppedIms.append(cropIm)

    return croppedIms

def format_pkl(all_pkl,camerasL,isGt,correctOffset,timestamp,fps_r):
    framesS03 = 'aic19-track1-mtmc-train/train/S03'
    allDetections = []
    boxes_List = []
    frames_List = []
    trackId_List = []
    cams_List = []
    for i,cam in enumerate(camerasL):
        data = []
        for j,id in enumerate(all_pkl[i]['id']):
            detections = []
            list_frames = all_pkl[i]['frame'][j]
            if len(np.where(np.array(list_frames)==-1)[0])>0:
                del list_frames[-1]
            for k,frame in enumerate(list_frames):
                boxes = all_pkl[i]['box'][j][k]
                cams_List.append(cam)
                trackId_List.append(id)
                frames_List.append(frame)
                boxes_List.append(boxes)
                if correctOffset:
                    frame = int(frame*fps_r[cam]+timestamp[cam])
                if not isGt:
                    frame_path = "{}/frames/{}.jpg".format(os.path.join(framesS03, cam), str(frame).zfill(5))
                    detections.append({'frame': frame,'frame_path':frame_path,'box': boxes})
                else:
                    detections.append({'frame':frame,'box':boxes})
            data.append({'track_id':id,'info':detections})

        allDetections.append({cam:data})
    frames_List, boxes_List, trackId_List, cams_List = zip(*sorted(zip(frames_List, boxes_List, trackId_List,cams_List)))

    return allDetections,list(cams_List),list(trackId_List),list(frames_List),list(boxes_List)

def compute_score(det_info):

    # init accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    ayuda = ['aic19-track1-mtmc-train/train/S03/c010/gt/gt.txt',
             'aic19-track1-mtmc-train/train/S03/c011/gt/gt.txt',
             'aic19-track1-mtmc-train/train/S03/c012/gt/gt.txt',
             'aic19-track1-mtmc-train/train/S03/c013/gt/gt.txt',
             'aic19-track1-mtmc-train/train/S03/c014/gt/gt.txt',
             'aic19-track1-mtmc-train/train/S03/c015/gt/gt.txt']

    ayuda2 = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']

    for cam_index in range(len(ayuda)):

        gt_info = NCA_train.import_gt_track(ayuda[cam_index])
        b = NCA_train.format_pkl([gt_info], [ayuda2[cam_index]])
        cams_List,trackId_List,frames_List,boxes_List = reformat_predictions(b)

        # gt correction
        cam_idx = np.where(np.array(cams_List) == ayuda2[cam_index])
        cam_idx = cam_idx[0]
        new_frames = [frames_List[id] for id in cam_idx]
        new_boxes = [boxes_List[id] for id in cam_idx]
        new_tracksId = [trackId_List[id] for id in cam_idx]

        # det correction
        cam_idx_det = np.where(np.array(det_info[0]) == ayuda2[cam_index])
        cam_idx_det = cam_idx_det[0]
        new_frames_det = [det_info[2][id] for id in cam_idx_det]
        new_boxes_det = [det_info[3][id] for id in cam_idx_det]
        new_tracksId_det = [det_info[1][id] for id in cam_idx_det]

        # Loop for all frames
        for frameID in trange(len(new_frames), desc="Score"):
            Nframe = new_frames[frameID]

            # get the ids of the tracks from the ground truth at this frame
            gt_list = [j for j, k in enumerate(new_frames) if k == Nframe]
            GTlist = [new_tracksId[i] for i in gt_list]
            GTbbox = [new_boxes[i] for i in gt_list]

            # get the ids of the detected tracks at this frame
            det_list = [j for j, k in enumerate(new_frames_det) if k == Nframe]
            DETlist = [new_tracksId_det[i] for i in det_list]
            DETbbox = [new_boxes_det[i] for i in det_list]

            # compute the distance for each pair
            distances = []
            for i in range(len(GTlist)):
                dist = []
                # compute the ground truth bbox
                bboxGT = GTbbox[i]
                # compute centroid GT
                centerGT = centroid(bboxGT)
                for j in range(len(DETlist)):
                    # compute the predicted bbox
                    bboxPR = DETbbox[j]
                    # compute centroid PR
                    centerPR = centroid(bboxPR)
                    d = euclid_dist(centerGT, centerPR)  # euclidean distance
                    dist.append(d)
                distances.append(dist)

            # update the accumulator
            acc.update(GTlist, DETlist, distances)

    # Compute and show the final metric results
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['idf1', 'idp', 'idr', 'precision', 'recall'], name='ACC:')
    strsummary = mm.io.render_summary(summary, formatters={'idf1': '{:.2%}'.format, 'idp': '{:.2%}'.format,
                                                           'idr': '{:.2%}'.format, 'precision': '{:.2%}'.format,
                                                           'recall': '{:.2%}'.format}, namemap={'idf1': 'IDF1',
                                                                                                'idp': 'IDP',
                                                                                                'idr': 'IDR',
                                                                                                'precision': 'Prec',
                                                                                                'recall': 'Rec'})
    print(strsummary)
