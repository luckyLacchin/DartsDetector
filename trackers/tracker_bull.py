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
    
    def interpolate_bull_positions(self, bull_positions, max_gap_frames=30):
        """
        Interpolates bull positions from the first detected frame up to max_gap_frames later.
        If it is detected again, a new interpolation starts from that frame.
        """
        extracted_positions = []
        for frame_idx, frame_data in enumerate(bull_positions):
            if 1 in frame_data and "bbox" in frame_data[1]:
                extracted_positions.append({"frame": frame_idx, "bbox": frame_data[1]["bbox"]})
        
        df_bull_positions = pd.DataFrame(extracted_positions)
        if df_bull_positions.empty:
            return bull_positions  # No detections, return original list
        
        df_bull_positions.set_index("frame", inplace=True)
        df_bull_positions = df_bull_positions.sort_index()
        
        # Group frames into separate sequences where gaps exceed max_gap_frames
        grouped_detections = []
        current_group = []
        prev_frame = None
        
        for _, row in df_bull_positions.iterrows():
            frame = row.name
            if prev_frame is not None and frame - prev_frame > max_gap_frames:
                grouped_detections.append(current_group)
                current_group = []
            current_group.append((frame, row["bbox"]))
            prev_frame = frame
        
        if current_group:
            grouped_detections.append(current_group)
        
        # Perform interpolation for each group separately
        for group in grouped_detections:
            frames, bboxes = zip(*group)
            df_group = pd.DataFrame(bboxes, index=frames, columns=["x1", "y1", "x2", "y2"])
            df_group = df_group.reindex(range(frames[0], min(frames[-1] + max_gap_frames, len(bull_positions))))
            df_group = df_group.interpolate().bfill()
            
            # Apply back to original list
            for frame, row in df_group.iterrows():
                if frame < len(bull_positions):
                    if 1 not in bull_positions[frame]:
                        bull_positions[frame][1] = {}
                    bull_positions[frame][1]["bbox"] = row.tolist()
        
        return bull_positions
    
   #Right now we are not using this function 
    def get_first_bull_center(self, tracks_bull):
        # Iterate through the list to find the first bull's center
        for frame_data in tracks_bull:
            if 'bull' in frame_data:  # Check if 'bull' data exists in the current frame
                bull_data = frame_data['bull']
                if bull_data and 1 in bull_data:  # Check if the bull data exists and contains the key '1'
                    bull_bbox = bull_data[1]['bbox']
                    return get_center_of_bbox(bull_bbox)

        return None  # Return None if no bull is found
    
    
    def get_latest_bull_center(self, tracks_bull, frame_num):
        # Iterate through the list up to the current frame number (inclusive)
        for frame_data in reversed(tracks_bull[:frame_num+1]):
            if 'bull' in frame_data:  # Check if 'bull' data exists in the current frame
                bull_data = frame_data['bull']
                if bull_data and 1 in bull_data:  # Check if the bull data exists and contains the key '1'
                    bull_bbox = bull_data[1]['bbox']
                    return get_center_of_bbox(bull_bbox)

        return None  # Return None if no bull is found



    
    '''
    def get_center_of_bull(self, tracks_bull, frame_num):
        # Iterate backwards through the frames in the list starting from frame_num
        for fn in range(frame_num, -1, -1):
            # Ensure that tracks_bull[fn] exists and has a 'bull' entry
            if len(tracks_bull) > fn and 'bull' in tracks_bull[fn]:
                bull_data = tracks_bull[fn]['bull']
                if bull_data and 1 in bull_data:  # Check if bull data exists and contains the key '1'
                    bull_bbox = bull_data[1]['bbox']  # Get the bounding box of the bull
                    return get_center_of_bbox(bull_bbox)

        # If no bull was detected in the previous frames, return the first detected bull
        first_bull_center = self.get_latest_bull_center(tracks_bull,frame_num)
        
        # If a first bull is found, return its center, else return None
        return first_bull_center if first_bull_center else None
    '''
    
    def get_center_of_bull(self, tracks_bull, frame_num):
        # Iterate backwards through the frames in the list starting from frame_num
        for fn in range(frame_num, -1, -1):
            # Ensure that tracks_bull[fn] exists and contains the key '1' (which represents bull data)
            if len(tracks_bull) > fn and 1 in tracks_bull[fn]:
                bull_data = tracks_bull[fn][1]
                bull_bbox = bull_data['bbox']
                return get_center_of_bbox(bull_bbox)

        # If no bull was detected in the previous frames, return the latest detected bull up to frame_num
        first_bull_center = self.get_latest_bull_center(tracks_bull, frame_num)
        
        # If a first bull is found, return its center, else return None
        return first_bull_center if first_bull_center else None
