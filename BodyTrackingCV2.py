# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 19:09:50 2023

@author: S. M . Hossein Mousavi
"""
# ----------------------------------------------------------------
import cv2
# ----------------------------------------------------------------
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MOSSE', 'CSRT']
# There are 6 main easy to use tracking algorihtms in cv2 of:
    # [0] : Boosting
    # [1] : Multiple Instance Learning (MIL)
    # [2] : Kernelized Correlation Filter (KCF)
    # [3] : Tracking, Learning and Detection (TLD)
    # [4] : Minimum Output Sum of Squared Error (MOSSE)
    # [5] : hannel and Spatial Reliability of discriminative correlation filter Tracker (CSRT)

# Help : you have to drag a rectangle by mouse and press enter.

# ----------------------------------------------------------------
# Select Tracker Below:
tracker_type = tracker_types[5] 
# ----------------------------------------------------------------
# BOOSTING is Based on AdaBoost algorithm [0]
if tracker_type == 'BOOSTING':
    tracker = cv2.legacy.TrackerBoosting_create()
# Multiple Instance Learning (MIL) [1]
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
# Kernelized Correlation Filter (KCF) utilizes adjacency matrix [2]
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create() 
# Tracking, Learning and Detection (TLD) [3]
if tracker_type == 'TLD':
    tracker = cv2.legacy.TrackerMedianFlow_create() 
# Minimum Output Sum of Squared Error (MOSSE) [4]
if tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
# Channel and Spatial Reliability of discriminative correlation filter Tracker (CSRT) [5]
if tracker_type == "CSRT": # The best
    tracker = cv2.TrackerCSRT_create()
# ----------------------------------------------------------------
# Load the Video File
video = cv2.VideoCapture("tst.mp4")
ret, frame = video.read()
# ----------------------------------------------------------------
frame_height, frame_width = frame.shape[:2]
# Decrease the video size for faster process
#frame = cv2.resize(frame, [frame_width//2, frame_height//2])
# Initialize video writer to save the results
output = cv2.VideoWriter(f'{tracker_type}.avi', 
                         cv2.VideoWriter_fourcc(*'XVID'), 60.0, 
                         (frame_width//1, frame_height//1), True)
# ----------------------------------------------------------------
if not ret:
    print('cannot read the video')
# Select the bounding box in the first frame
bbox = cv2.selectROI(frame, False)
# Initialize the tracker 
ret = tracker.init(frame, bbox)
# Start tracking (frame by frame processing)
while True:
    ret, frame = video.read()
    #frame = cv2.resize(frame, [frame_width//2, frame_height//2])
    if not ret:
        print('something went wrong') 
        break
    timer = cv2.getTickCount()
    ret, bbox = tracker.update(frame)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100,80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    cv2.putText(frame, tracker_type + " Tracker", (100,20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
    cv2.imshow("Tracking", frame)
    output.write(frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
# ----------------------------------------------------------------
video.release()
output.release()
cv2.destroyAllWindows()