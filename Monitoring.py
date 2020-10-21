# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:15:06 2020

@author: kjosh
"""
from tensorflow.keras.models import load_model
from scipy.spatial import distance as dist
from Mask_distance import detect_people,detect_masked_people
import imutils
import numpy as np
import cv2



weightsPath = 'yolov3.weights'
configPath = 'yolov3.cfg'
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
labelsPath = 'coco.names'
LABELS = open(labelsPath).read().strip().split("\n")



prototxtPath = 'deploy.prototxt'
weightsPath_face ='res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath_face)
maskNet = load_model('labels_mask.model')


ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture('Video3.mp4')
writer = None



while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    frame = imutils.resize(frame, width=1000)
    results = detect_people(frame, net, ln,personIdx=LABELS.index("person"))
    (locs, preds) = detect_masked_people(frame, faceNet, maskNet)
    
    violate = set()
    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")
        
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                if D[i, j] < 50:
                    violate.add(i)
                    violate.add(j)
    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)
        if i in violate:
            color = (0, 0, 255)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        #cv2.circle(frame, (cX, cY), 5, color, 1)
    
    for (box1, pred) in zip(locs, preds):
        (startX1, startY1, endX1, endY1) = box1
        (withoutMask,mask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (255, 255, 0) if label == "Mask" else (255, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (startX1, startY1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX1, startY1), (endX1, endY1), color, 2)
        

    text = "Distancing Violations: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_TRIPLEX , 0.7, (0, 0, 255), 3)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        writer = cv2.VideoWriter('output.avi', fourcc, 10,(frame.shape[1], frame.shape[0]))
    
    if writer is not None:
        writer.write(frame)

    
cv2.destroyAllWindows()