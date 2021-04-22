from tqdm import trange
import numpy as np
import pickle
import cv2
import motmetrics as mm

from dataReader import ReadData
from utils import *
import NCA_train
from histograms import *
from noGridSearch import *


def task1():
    # Paths
    det_path = 'aic19-track1-mtmc-train/train/S03/c011/det/det_yolo3.txt'  # 'retinanet_S03/prediction_results_retina_S03_c010.pkl'
    video_path = 'aic19-track1-mtmc-train/train/S03/c011/vdo.avi'
    gt_path = 'aic19-track1-mtmc-train/train/S03/c011/gt/gt.txt'

    # Parameters
    threshold = 0.5  # minimum iou to consider the tracking between consecutive frames
    kill_time = 90  # nº of frames to close the track of an object
    presence = 20  # nº of frames required of an object to exist
    movement = 150  # minimum distance that a car requires to do to be considered (from start to end)
    small_movement = 10  # minimum distance that a car requires to do to be considered (each 10 frames)
    min_size = 1000000  # minimum area of the bbox to be detected

    # Flags
    video = False
    showVid = False
    showGT = True
    showDET = True
    compute_score = True
    save_tracks = True
    optical_flow = False

    # Read the detections from files
    frame_bboxes = []
    new_frame_bboxes = []
    if det_path[-3:] == 'pkl':
        # Get the bboxes (Read from our pickles)
        with (open(det_path, "rb")) as openfile:
            while True:
                try:
                    frame_bboxes.append(pickle.load(openfile))
                except EOFError:
                    break
        frame_bboxes = frame_bboxes[0]
        # correct the data to the desired format
        for frame_index in range(frame_bboxes[0][-1] + 1):
            position_index = [j for j, k in enumerate(frame_bboxes[0]) if k == frame_index]
            if len(position_index) == 0:
                new_frame_bboxes.append([])
            else:
                aux_list = []
                for superindex in position_index:
                    aux_bbox = frame_bboxes[3][superindex].cpu().numpy()
                    aux_list.append(aux_bbox)
                new_frame_bboxes.append(aux_list)
    elif det_path[-3:] == 'txt':
        # Load detections (Read from the dataset)
        readerDET = ReadData(det_path)
        frame_bboxes = readerDET.getDETfromTXT()[0]
        # transform det
        for frame_index in range(frame_bboxes[-1][0] + 1):
            position_index = [j for j, k in enumerate(frame_bboxes) if k[0] == frame_index]
            if len(position_index) == 0:
                new_frame_bboxes.append([])
            else:
                aux_list = []
                for superindex in position_index:
                    aux_bbox = np.array(frame_bboxes[superindex][3:7])
                    aux_list.append(aux_bbox)
                new_frame_bboxes.append(aux_list)

    # Once we have done the detection we can start with the tracking
    cap = cv2.VideoCapture(video_path)
    previous_frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
    bbox_per_frame = []
    id_per_frame = []
    frame = new_frame_bboxes[0]  # load the bbox for the first frame
    # Since we evaluate the current frame and the consecutive, we loop for range - 1
    for Nframe in trange(len(new_frame_bboxes) - 1, desc="Tracking"):
        next_frame = new_frame_bboxes[Nframe + 1]
        current_frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)

        if optical_flow:
            # apply optical flow to improve the bounding box and get better iou with the following frame
            # predict flow with block matching
            blockSize = 16
            searchArea = 96
            quantStep = 16
            method = 'cv2.TM_CCORR_NORMED'
            predicted_flow = compute_block_matching(previous_frame, current_frame, 'backward', searchArea, blockSize,
                                                    method, quantStep)

        # assign a new ID to each unassigned bbox
        for i in range(len(frame)):
            new_bbox = frame[i]
            area = new_bbox[2] * new_bbox[3]
            if area < min_size:
                continue

            # append the bbox to the list
            bbox_per_id = []
            bbox_per_id.append(list(new_bbox))
            bbox_per_frame.append(bbox_per_id)
            # append the id to the list
            index_per_id = []
            index_per_id.append(Nframe)
            id_per_frame.append(index_per_id)

        # we loop for each track and we compute the iou with each detection of the next frame
        for id in range(len(bbox_per_frame)):
            length = len(bbox_per_frame[id])
            bbox_per_id = bbox_per_frame[id]  # bboxes of a track
            bbox1 = bbox_per_id[length - 1]  # last bbox stored of the track
            index_per_id = id_per_frame[id]  # list of frames where the track appears

            if optical_flow:
                vectorU = predicted_flow[int(bbox1[1]):int(bbox1[3]), int(bbox1[0]):int(bbox1[2]), 0]
                vectorV = predicted_flow[int(bbox1[1]):int(bbox1[3]), int(bbox1[0]):int(bbox1[2]), 1]
                dx = vectorU.mean()
                dy = vectorV.mean()
                # apply movemement to the bbox
                new_bbox1 = list(np.zeros(4))
                new_bbox1[0] = bbox1[0] + dx
                new_bbox1[2] = bbox1[2] + dx
                new_bbox1[1] = bbox1[1] + dy
                new_bbox1[3] = bbox1[3] + dy

            # don't do anything if the track is closed
            if index_per_id[-1] == -1:
                continue

            # get the list of ious, one with each detection of the next frame
            iou_list = []
            for detections in range(len(next_frame)):
                if detections >= len(next_frame):
                    break
                bbox2 = next_frame[detections]  # detection of the next frame
                area = bbox2[2] * bbox2[3]
                if area < min_size:
                    iou_list.append(0)  # we fake a low iou
                    continue
                if optical_flow:
                    iou_list.append(iou(np.array(new_bbox1), bbox2))
                else:
                    iou_list.append(iou(np.array(bbox1), bbox2))

            # break the loop if there are no more bboxes in the frame to track
            if len(next_frame) == 0:
                # kill_time control
                not_in_scene = Nframe - index_per_id[-1]  # nº of frames that we don't track this object
                if not_in_scene > kill_time:  # if it surpasses the kill_time, close the track by adding a -1
                    index_per_id.append(-1)
                break

            # assign the bbox to the closest track
            best_iou = max(iou_list)
            # if the mas iou is lower than 0.5, we assume that it doesn't have a correspondence
            if best_iou > threshold:
                best_detection = [j for j, k in enumerate(iou_list) if k == best_iou]
                best_detection = best_detection[0]

                # append to the list the bbox of the next frame
                bbox_per_id.append(list(next_frame[best_detection]))
                index_per_id.append(Nframe + 1)

                # we delete the detection from the list in order to speed up the following comparisons
                del next_frame[best_detection]
            else:
                # kill_time control
                not_in_scene = Nframe - index_per_id[-1]  # nº of frames that we don't track this object
                if not_in_scene > kill_time:  # if it surpasses the kill_time, close the track by adding a -1
                    index_per_id.append(-1)

        frame = next_frame  # the next frame will be the current
        previous_frame = current_frame  # update the frame for next iteration

    # Post-processing
    for track in trange(len(bbox_per_frame), desc="Post-processing"):
        # delete the tracks that does not exists for a minimum required time
        if len(bbox_per_frame[track]) < presence:
            bbox_per_frame[track] = None
            id_per_frame[track] = None
        else:
            # delete the tracks of the parked cars
            bbox1 = bbox_per_frame[track][0]
            bbox2 = bbox_per_frame[track][-1]
            centerBbox1 = centroid(bbox1)
            centerBbox2 = centroid(bbox2)
            dist = euclid_dist(centerBbox1, centerBbox2)  # euclidean distance
            # delete if the bbox did not move a required minimum distance
            if dist < movement:
                bbox_per_frame[track] = None
                id_per_frame[track] = None
            # cut the detections while the car still has not move
            else:
                # check the movement each 10 frames, 1 second
                for frame_jump in range(10, len(bbox_per_frame[track]), 10):
                    bbox1 = bbox_per_frame[track][frame_jump - 10]
                    bbox2 = bbox_per_frame[track][frame_jump]
                    centerBbox1 = centroid(bbox1)
                    centerBbox2 = centroid(bbox2)
                    dist = euclid_dist(centerBbox1, centerBbox2)
                    if dist > small_movement:  # car starts moving
                        bbox_per_frame[track] = bbox_per_frame[track][frame_jump - 10:]
                        id_per_frame[track] = id_per_frame[track][frame_jump - 10:]
                        break
    # delete the Nones
    bbox_per_frame = [i for i in bbox_per_frame if i]
    id_per_frame = [i for i in id_per_frame if i]

    if video:
        # Generate colors for each track
        id_colors = []
        for i in range(len(id_per_frame)):
            color = list(np.random.choice(range(256), size=3))
            id_colors.append(color)

        # Load gt for plot
        reader = ReadData(gt_path)
        gt, num_iter = reader.getGTfromTXT()

        # Define the codec and create VideoWriter object
        vidCapture = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('task.avi', fourcc, 8.0, (1920, 1080))
        # for each frame draw rectangles to the detected bboxes
        for i in trange(len(new_frame_bboxes), desc="Video"):
            vidCapture.set(cv2.CAP_PROP_POS_FRAMES, i)
            im = vidCapture.read()[1]
            # draw gt
            GTdet_in_frame = [j for j, k in enumerate(gt) if k[0] == i]
            if showGT:
                for gtDet in GTdet_in_frame:
                    bbox = gt[gtDet][3:7]
                    id = gt[gtDet][1]
                    cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                    cv2.putText(im, 'ID: ' + str(id) + ' (GT)', (int(bbox[0]), int(bbox[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            # draw detections
            if showDET:
                for id in range(len(id_per_frame)):
                    ids = id_per_frame[id]
                    if i in ids:
                        id_index = ids.index(i)
                        bbox = bbox_per_frame[id][id_index]
                        color = id_colors[id]
                        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                      (int(color[0]), int(color[1]), int(color[2])), 2)
                        cv2.putText(im, 'ID: ' + str(id), (int(bbox[0]), int(bbox[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (int(color[0]), int(color[1]), int(color[2])), 2)
            if showVid:
                cv2.imshow('Video', im)
            out.write(im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vidCapture.release()
        out.release()
        cv2.destroyAllWindows()

    if save_tracks:
        filename = 'tracks_c10.pkl'
        dictionary = {}
        dictionary['id'] = list(range(len(bbox_per_frame)))
        dictionary['frame'] = id_per_frame
        dictionary['box'] = bbox_per_frame
        outfile = open(filename, 'wb')
        pickle.dump(dictionary, outfile)
        outfile.close()

    if compute_score:
        # Load gt for plot
        reader = ReadData(gt_path)
        gt, num_iter = reader.getGTfromTXT()

        # init accumulator
        acc = mm.MOTAccumulator(auto_id=True)

        # Loop for all frames
        for Nframe in trange(len(new_frame_bboxes), desc="Score"):

            # get the ids of the tracks from the ground truth at this frame
            gt_list = [item[1] for item in gt if item[0] == Nframe]
            gt_list = np.unique(gt_list)

            # get the ids of the detected tracks at this frame
            pred_list = []
            for ID in range(len(id_per_frame)):
                aux = np.where(np.array(id_per_frame[ID]) == Nframe)[0]
                if len(aux) > 0:
                    pred_list.append(int(ID))

            # compute the distance for each pair
            distances = []
            for i in range(len(gt_list)):
                dist = []
                # compute the ground truth bbox
                bboxGT = gt_list[i]
                bboxGT = [item[3:7] for item in gt if (item[0] == Nframe and item[1] == bboxGT)]
                bboxGT = list(bboxGT[0])
                # compute centroid GT
                centerGT = centroid(bboxGT)
                for j in range(len(pred_list)):
                    # compute the predicted bbox
                    bboxPR = pred_list[j]
                    aux_id = id_per_frame[bboxPR].index(Nframe)
                    bboxPR = bbox_per_frame[bboxPR][aux_id]
                    # compute centroid PR
                    centerPR = centroid(bboxPR)
                    d = euclid_dist(centerGT, centerPR)  # euclidean distance
                    dist.append(d)
                distances.append(dist)

            # update the accumulator
            acc.update(gt_list, pred_list, distances)

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

def task2():
    # Relative paths
    gt_paths = ['aic19-track1-mtmc-train/train/S01', 'aic19-track1-mtmc-train/train/S04']
    fps_r = {
        'c010': 1.0,
        'c011': 1.0,
        'c012': 1.0,
        'c013': 1.0,
        'c014': 1.0,
        'c015': 10.0 / 8.0,
    }
    timestamp = {
        'c010': 8.715,
        'c011': 8.457,
        'c012': 5.879,
        'c013': 0,
        'c014': 5.042,
        'c015': 8.492,
    }
    base = 'aic19-track1-mtmc-train/train/S03'
    video_path = {
        'c010': "{}/c010/vdo.avi".format(base),
        'c011':  "{}/c011/vdo.avi".format(base),
        'c012':  "{}/c012/vdo.avi".format(base),
        'c013':  "{}/c013/vdo.avi".format(base),
        'c014':  "{}/c014/vdo.avi".format(base),
        'c015': "{}/c015/vdo.avi".format(base),
    }
    frame_size = {
        'c010': [1920, 1080],
        'c011': [2560,1920],
        'c012': [2560, 1920],
        'c013': [2560, 1920],
        'c014': [ 1920, 1080],
        'c015': [ 1920, 1080]
    }

    camerasL = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    pickle_names = ['c10', 'c11', 'c12', 'c13', 'c14', 'c15']

    # Pickle paths (to speed the experiments)
    path_model = "mr_hsv_s01s03.pkl"
    path_multicamera_det = "correctDet_hsvS04_thr50.pkl"

    # Flags
    train = False
    compute_test = False
    video = False
    showVid = False

    # Load tracking detections
    dets = []
    for pickle_name in pickle_names:
        file = "tracks_"+ pickle_name +".pkl"
        with open(file, 'rb') as f:
            dets.append(pickle.load(f))
            f.close()

    detections,_,_,_,_ = format_pkl(all_pkl=dets, camerasL=camerasL, isGt=False, correctOffset=False,
                                    timestamp=timestamp, fps_r=fps_r)

    if train:
        gtdata, vid_paths = NCA_train.get_gt_info(gt_paths)
        nca_classif = NCA_train.train_NCA(gtdata, vid_paths)
    else:
        print('LOAD PKL')
        with open(path_model, 'rb') as f:
            nca_classif = pickle.load(f)
            f.close()

    print('TEST...')
    index_box_track = [0.25, 0.5, 0.60]
    count_cams = 1
    total_scores = []
    corrected_detections = detections

    if compute_test:
        for idCam in trange(len(detections)-1):
            cam1 = camerasL[idCam]
            cam2 = camerasL[count_cams]
            count_cams += 1
            num_tracks1 = len(detections[idCam][cam1])
            num_tracks2 = len(detections[idCam+1][cam2])
            for i in range(num_tracks1):
                track1 = detections[idCam][cam1][i]
                cropped_bboxes_cam1 = crop_image(track1,index_box_track)
                for j in range(num_tracks2):
                    track2 = detections[idCam+1][cam2][j]
                    cropped_bboxes_cam2 = crop_image(track2,index_box_track)
                    scores = []
                    for n in range(len(cropped_bboxes_cam1)):
                        ft_vecCam1 = histogram_multires(cropped_bboxes_cam1[n])
                        ft_vecCam2 = histogram_multires(cropped_bboxes_cam2[n])
                        formPairs = [np.vstack((ft_vecCam1,ft_vecCam2))]
                        formPairs= np.array(formPairs)
                        score = nca_classif.score_pairs(formPairs).mean()
                        scores.append(score)
                    mean_score = np.mean(np.array(scores))
                    total_scores.append(scores)
                    if mean_score < 50:
                        track2['track_id'] = track1['track_id']
                        # Update the detections
                        corrected_detections[idCam][cam1][i] = track1
                        corrected_detections[idCam+1][cam2][j] = track2
    else:
        print('LOAD PKL')
        with open(path_multicamera_det, 'rb') as f:
            corrected_detections = pickle.load(f)
            f.close()
    cams_List,trackId_List,frames_List,boxes_List = reformat_predictions(corrected_detections)

    if video:
        # Generate colors for each track
        id_colors = []
        length = []
        for cc in range(len(camerasL)):
            length.append(len(corrected_detections[cc][camerasL[cc]]))
        max_tracks = np.max(length)

        for i in range(max_tracks):
            color = list(np.random.choice(range(256), size=3))
            id_colors.append(color)
        for cam in camerasL:
            # Define the codec and create VideoWriter object
            vidCapture = cv2.VideoCapture(video_path[cam])
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(cam+'_task2.avi', fourcc, 10.0, (frame_size[cam][0], frame_size[cam][1]))
            # filter the cam detections
            cam_idx = np.where(np.array(cams_List) == cam)
            cam_idx = cam_idx[0]
            new_frames = [frames_List[id] for id in cam_idx]
            new_boxes = [boxes_List[id] for id in cam_idx]
            new_tracksId = [trackId_List[id] for id in cam_idx]
            # for each frame draw rectangles to the detected bboxes
            for i,fr in enumerate(np.unique(new_frames)):
                idsFr = np.where(np.array(new_frames)==fr)
                idsFr = idsFr[0]
                vidCapture.set(cv2.CAP_PROP_POS_FRAMES, fr)
                im = vidCapture.read()[1]

                for j in range(len(idsFr)):
                    track_id = new_tracksId[idsFr[j]]
                    bbox =  new_boxes[idsFr[j]]
                    color = id_colors[track_id]
                    cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                  (int(color[0]), int(color[1]), int(color[2])), 2)
                    cv2.putText(im, 'ID: ' + str(track_id), (int(bbox[0]), int(bbox[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (int(color[0]), int(color[1]), int(color[2])), 2)

                if showVid:
                    cv2.imshow('Video', im)
                out.write(im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            vidCapture.release()
            out.release()
            cv2.destroyAllWindows()

    det_info = reformat_predictions(corrected_detections)
    compute_score(det_info)


if __name__ == '__main__':
    #task1()
    task2()