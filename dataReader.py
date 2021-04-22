import json
import xmltodict
import numpy as np


def _information(frame, id, label, xtl, ytl, xbr, ybr, score=None, parked=None):

    return frame,id,label,xtl,ytl,xbr,ybr,score,parked


class ReadData:
    def __init__(self,path):
        self.path = path

    def getDETfromTXT(self):
        path = self.path

        with open(path) as f:
            lines = f.readlines()

        det = []
        num_iter = 0
        for line in lines:
            spt_line = line.split(',')
            det.append(_information(
                frame=int(spt_line[0]) - 1,
                id=int(-1),
                label='car',
                xtl=float(spt_line[2]),
                ytl=float(spt_line[3]),
                xbr=float(spt_line[2]) + float(spt_line[4]),
                ybr=float(spt_line[3]) + float(spt_line[5]),
                score=float(spt_line[6])
            ))
            num_iter += 1

        return det, num_iter

    def getGTfromXML(self):
        path = self.path

        with open(path) as f:
            tracks = xmltodict.parse(f.read())['annotations']['track']

        gt = []
        num_iter = 0
        for track in tracks:
            id = track['@id']
            label = track['@label']
            boxes = track['box']
            for box in boxes:
                #remove this if  if you want to load bikes
                if label == 'car':
                    if label == 'car':
                        parked = box['attribute']['#text'].lower() == 'true'
                    else:
                        parked = None
                    gt.append(_information(
                        frame=int(box['@frame']),
                        id=int(id),
                        label=label,
                        xtl=float(box['@xtl']),
                        ytl=float(box['@ytl']),
                        xbr=float(box['@xbr']),
                        ybr=float(box['@ybr']),
                        parked=parked
                    ))
                    num_iter += 1

        return gt, num_iter

    #Used to remove the parked cars
    def preprocessGT(self,gt):
        newGT = gt
        for i in range(len(gt)):
            g = gt[i]
            newGT[i] = list(newGT[i])
            if g[-1] == True:
                newGT[i][3] = -1
                newGT[i][4] = -1
                newGT[i][5] = -1
                newGT[i][6] = -1
        return newGT

    # Extract boxes per frame
    def bboxInFrame(self,gt,initFrame):
        frames = []
        BBOX = []
        for infos in gt:
            frame = infos[0]
            if frame >= initFrame:
                if infos[3] == -1:
                    bbox = [-1]

                else:
                    bbox = [infos[3],infos[4],infos[5],infos[6]]
                frames.append(frame)
                BBOX.append(bbox)

        # Sort frames
        sortedFrames, sortedBbox = zip(*sorted(zip(frames, BBOX)))

        return sortedFrames, sortedBbox,len(sortedBbox)

    # Extract boxes and scores per frame
    def bboxInFrame_Score(self, pred):
        frames = []
        BBOX = []
        score_val = []
        for infos in pred:
            frame = infos[0]
            bbox = [infos[3], infos[4], infos[5], infos[6]]
            score = infos[7]
            frames.append(frame)
            BBOX.append(bbox)
            score_val.append(score)

        # Sort frames
        sortedFrames, sortedBbox, sortedScore = zip(*sorted(zip(frames, BBOX, score_val)))

        return sortedFrames, sortedBbox, sortedScore

    def joinBBOXfromFrame(self,sortedFrames,sortedBbox,isGT):
        bbox = []
        Info = []
        # numb = 0
        # sortedBbox = [[1,1,1,1],None,[3,3,3,3],[4,4,4,4],[5,5,5,5],[6,6,6,6],[7,7,7,7]]
        # sortedFrames = [0,0,1,2,3,3,3]
        sortedBbox = list(sortedBbox)
        for i in range(len(sortedBbox)):
            if i ==  0:
                if sortedBbox[i] == [-1]:
                    sortedBbox[i] = None
                bbox.append(sortedBbox[i])

            else:
                if sortedFrames[i] == sortedFrames[i-1]:
                    if sortedBbox[i] == [-1]:
                        sortedBbox[i] = None
                    bbox.append(sortedBbox[i])
                else:
                    if isGT:
                        is_detected = [False]*len(bbox)
                        Info.append({"frame": sortedFrames[i-1],"bbox": np.array(bbox), "is_detected": is_detected})
                    else:
                        Info.append({"frame": sortedFrames[i - 1], "bbox": np.array(bbox)})
                    bbox = []
                    if sortedBbox[i] == [-1]:
                        sortedBbox[i] = None
                    bbox.append(sortedBbox[i])

                if i+1 == len(sortedBbox):
                    if isGT:
                        is_detected = [False] * len(bbox)
                        Info.append({"frame": sortedFrames[i - 1], "bbox": np.array(bbox), "is_detected": is_detected})
                    else:
                        Info.append({"frame": sortedFrames[i - 1], "bbox": np.array(bbox)})

        return Info

    def fixFormat(self,sortedFrames,sortedBbox,sortedlabels,sortedscores,isGT):
        bbox = []
        Info = []
        label = []
        score = []
        sortedBbox = list(sortedBbox)
        for i in range(len(sortedBbox)):
            if i ==  0:
                if sortedBbox[i] == [-1]:
                    sortedBbox[i] = None
                bbox.append(sortedBbox[i])
                label.append(sortedlabels[i])
                score.append(sortedscores[i])

            else:
                if sortedFrames[i] == sortedFrames[i-1]:
                    if sortedBbox[i] == [-1]:
                        sortedBbox[i] = None
                    bbox.append(sortedBbox[i])
                    label.append(sortedlabels[i])
                    score.append(sortedscores[i])
                else:
                    if isGT:
                        is_detected = [False]*len(bbox)
                        Info.append({"frame": sortedFrames[i-1],"label": label, "bbox": bbox, "score": score, "is_detected": is_detected})
                    else:
                        Info.append({"frame": sortedFrames[i - 1], "label":label, "bbox": bbox, "score":score})
                    bbox = []
                    if sortedBbox[i] == [-1]:
                        sortedBbox[i] = None
                    bbox.append(sortedBbox[i])
                    label.append(sortedlabels[i])
                    score.append(sortedscores[i])

                if i+1 == len(sortedBbox):
                    if isGT:
                        is_detected = [False] * len(bbox)
                        Info.append({"frame": sortedFrames[i-1],"label": label, "bbox": bbox, "score": score, "is_detected": is_detected})
                    else:
                        Info.append({"frame": sortedFrames[i - 1], "label":label, "bbox": bbox, "score":score})

        return Info

    def resetGT(self,gt):
        for i in range(len(gt)):
            num = len(gt[i]['is_detected'])
            gt[i]['is_detected'] = [False] * num

        return gt

    def getPredfromTXT(self):
        # Prediction format  [frame,-1,left,top,width,height,conf,-1,-1,-1]
        path = self.path

        with open(path) as f:
            lines = f.readlines()

        pred = []
        num_iter = 0
        for line in lines:
            spt_line = line.split(',')
            pred.append(_information(
                frame=int(spt_line[0]) - 1,
                id=spt_line[1],
                label='car',
                xtl=float(spt_line[2]),
                ytl=float(spt_line[3]),
                xbr=float(spt_line[2]) + float(spt_line[4]),
                ybr=float(spt_line[3]) + float(spt_line[5]),
                score=float(spt_line[6])
            ))
            num_iter += 1

        return pred, num_iter

    def getGTfromTXT(self):
        path = self.path

        with open(path) as f:
            lines = f.readlines()

        gt = []
        num_iter = 0
        for line in lines:
            spt_line = line.split(',')
            gt.append(_information(
                frame=int(spt_line[0]) - 1,
                id=int(spt_line[1]),
                label='car',
                xtl=float(spt_line[2]),
                ytl=float(spt_line[3]),
                xbr=float(spt_line[2]) + float(spt_line[4]),
                ybr=float(spt_line[3]) + float(spt_line[5]),
                score=float(-1)
            ))
            num_iter += 1

        return gt, num_iter

    def joinBBOXfromFrame_Score(self,sortedFrames,sortedBbox,sortedScore,isGT):
        bbox = []
        score = []
        Info = []
        for i in range(len(sortedBbox)):
            if i ==  0:
                bbox.append(sortedBbox[i])
                score.append(sortedScore[i])

            else:
                if sortedFrames[i] == sortedFrames[i-1]:
                    bbox.append(sortedBbox[i])
                    score.append(sortedScore[i])
                else:
                    if isGT:
                        is_detected = [False]*len(bbox)
                        Info.append({"frame": sortedFrames[i-1],"bbox": np.array(bbox), "score": np.array(score), "is_detected": is_detected})
                    else:
                        Info.append({"frame": sortedFrames[i - 1], "bbox": np.array(bbox), "score": np.array(score)})
                    bbox = []
                    score = []
                    bbox.append(sortedBbox[i])
                    score.append(sortedScore[i])

                if i+1 == len(sortedBbox):
                    if isGT:
                        is_detected = [False] * len(bbox)
                        Info.append({"frame": sortedFrames[i - 1], "bbox": np.array(bbox), "score": np.array(score), "is_detected": is_detected})
                    else:
                        Info.append({"frame": sortedFrames[i - 1], "bbox": np.array(bbox), "score": np.array(score)})

        return Info


