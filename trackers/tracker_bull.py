from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
sys.path.append("../")
from utils import get_center_of_bbox, get_bbox_width, draw_rectangle
import cv2
import numpy as np
import pandas as pd
import math

class TrackerBull:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.4) 
            detections += detections_batch

        return detections
    
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        detections = self.detect_frames(frames)

        tracks=[]

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            tracks.append({})
            chosen_bbox =None
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv['bull']:
                    chosen_bbox = bbox
            
            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox":chosen_bbox}

        
        return tracks
    
    
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            bull_dict = tracks[frame_num]

            # Draw Bull 
            for _, bull in bull_dict.items():
                frame = draw_rectangle(frame, bull["bbox"],label_text="Bull",color=(255,255,0))

            output_video_frames.append(frame)
            
        return output_video_frames

