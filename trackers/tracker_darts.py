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

class TrackerDarts:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()
        self.last_interpolated_darts = []  # To keep track of the last 3 interpolated darts
    
    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.4) #maybe here i can write also a lower conf
            detections += detections_batch

        return detections


    def interpolate_darts_positions(self, tracks):
        """
        Interpolates dart tip positions using tracks["linked_darts"] (which is a list of frames).
        Interpolates for each dart from the first frame it appears to the last frame it appears.
        """
        linked_darts = tracks["linked_darts"]  # Extract linked darts from tracks
        frame_keys = range(len(linked_darts))  # Create a range of frame indices (since it's a list)

        all_darts = []

        # Extract dart positions frame by frame
        for frame_num in frame_keys:
            frame_data = linked_darts[frame_num]
            for track_id, data in frame_data.items():
                if "bbox" in data:
                    x1, y1, x2, y2 = data["bbox"]
                    all_darts.append({"frame": frame_num, "track_id": track_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

        #print("all darts: ", all_darts)
        # Convert list to DataFrame
        df_darts = pd.DataFrame(all_darts, columns=["frame", "track_id", "x1", "y1", "x2", "y2"])
        #print(df_darts)
        #print(f"items(): {df_darts.items()}")


        # Interpolate for each dart from its first appearance to its last frame
        for track_id in df_darts["track_id"].unique():
            dart_data = df_darts[df_darts["track_id"] == track_id]
            #print(dart_data)
            if dart_data.empty:
                continue
            
            first_frame = dart_data["frame"].min()
            last_frame = dart_data["frame"].max()
            
            first_frame = int(first_frame)
            last_frame = int(last_frame)
            '''
            print(f"first: {first_frame}")
            print(f"last_frame: {last_frame}")
            '''
            # Interpolate from first to last frame
            dart_data_interpolated = dart_data[dart_data["frame"] <= last_frame]
            dart_data_interpolated = dart_data_interpolated.set_index("frame").reindex(range(first_frame, last_frame + 1))
            dart_data_interpolated = dart_data_interpolated.interpolate().reset_index()

            #print(dart_data_interpolated)
            
            df_darts = pd.concat([df_darts, dart_data_interpolated], ignore_index=True).drop_duplicates()

        #print(df_darts)

        # Convert back to dictionary format
        interpolated_darts = {}
        #print("interpolated: ", interpolated_darts)
        for _, row in df_darts.iterrows():
            frame = int(row["frame"])
            track_id = int(row["track_id"])
            
            
            if track_id not in interpolated_darts:
                interpolated_darts[track_id] = {} 
            
            interpolated_darts[track_id][frame] = {
                "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]]
            }

        return interpolated_darts



    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Process the frames, detect objects, and track their positions.
        Link dart components together while handling missing detections and merging virtual darts with real ones.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        else:
            detections = self.detect_frames(frames)
            min_conf_board = 0.4
            min_conf = 0.8
        
        
            tracks = {
                "board": [],
                "darts": [],  # Full dart detections
                "tips": [],
                "barrels": [],
                "shafts": [],
                "flights": [],
                "linked_darts": [],  # Linked dart components
                "virtual_darts": {}  # Virtual darts to link unpaired parts
            }

            all_detected_parts = []
            detected_darts = {}

            # Process each frame and detect parts
            for frame_num, detection in enumerate(detections):
                cls_names = detection.names
                cls_names_inv = {v: k for k, v in cls_names.items()}

                detection_supervision = sv.Detections.from_ultralytics(detection)
                detections_with_tracks = self.tracker.update_with_detections(detection_supervision)

                # Initialize tracking dictionaries for the current frame
                tracks["darts"].append({})
                tracks["tips"].append({})
                tracks["barrels"].append({})
                tracks["shafts"].append({})
                tracks["flights"].append({})
                tracks["linked_darts"].append({})
                tracks["board"].append({})
                
                detected_parts = []

                # Step 1: Detect parts and link to darts (if possible)
                for frame_detection in detections_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]
                    conf = frame_detection[2]

                    if cls_id == cls_names_inv["Dart"] and conf > min_conf:
                        detected_darts[track_id] = {"bbox": bbox, "parts": {"tip": None, "barrel": None, "shaft": None, "flight": None}, "frame_num": frame_num}
                        tracks["darts"][frame_num][track_id] = {"bbox": bbox}

                    elif cls_id == cls_names_inv["Tip"] and conf > min_conf:
                        detected_parts.append(("tip", track_id, bbox))
                        tracks["tips"][frame_num][track_id] = {"bbox": bbox}

                    elif cls_id == cls_names_inv["Barrel"] and conf > min_conf: 
                        detected_parts.append(("barrel", track_id, bbox))
                        tracks["barrels"][frame_num][track_id] = {"bbox": bbox}

                    elif cls_id == cls_names_inv["Shaft"] and conf > min_conf:
                        detected_parts.append(("shaft", track_id, bbox))
                        tracks["shafts"][frame_num][track_id] = {"bbox": bbox}

                    elif cls_id == cls_names_inv["Flight"] and conf > min_conf:
                        detected_parts.append(("flight", track_id, bbox))
                        tracks["flights"][frame_num][track_id] = {"bbox": bbox}
                
                # Tracking for the board    
                for frame_detection in detection_supervision:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]

                    if cls_id == cls_names_inv['Board'] and conf > min_conf_board:
                        tracks["board"][frame_num][1] = {"bbox":bbox}

                all_detected_parts.append((frame_num, detected_parts))

            # Step 2: Associate parts to darts after all frames have been processed
            for frame_num, detected_parts in all_detected_parts:
                for part_name, part_id, part_bbox in detected_parts:
                    linked_dart_id = self.find_closest_dart(part_bbox, detected_darts, frame_num)

                    if linked_dart_id is not None:
                        # Link part to an existing dart
                        if detected_darts[linked_dart_id]["parts"][part_name] is None:
                            detected_darts[linked_dart_id]["parts"][part_name] = part_id
                            dart_data_copy = {key: value for key, value in detected_darts[linked_dart_id].items() if key != 'frame_num'}
                            tracks["linked_darts"][frame_num][linked_dart_id] = dart_data_copy
                        else:
                            # Part is already linked, check for virtual darts
                            self._link_part_to_virtual_dart(part_id, part_name, part_bbox, tracks, frame_num, detected_darts)
                            

                    else:
                        # Part isn't linked to any dart, create a virtual dart
                        self._link_part_to_virtual_dart(part_id, part_name, part_bbox, tracks, frame_num, detected_darts)
                    
        
            if stub_path is not None:
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracks, f)
            
            '''
            #print("tracks[linked_darts]", tracks["linked_darts"])
            for frame_num, linked_darts_data in enumerate(tracks["linked_darts"]):
                #print(f"Starting linked_darts for frame {frame_num}:")
                #print(linked_darts_data)
                if frame_num == 60:
                    break
            '''
            return tracks

    def _link_part_to_virtual_dart(self, part_id, part_name, part_bbox, tracks, frame_num, detected_darts):
        """
        Helper function to link a part to a virtual dart if not already linked.
        """
        part_already_linked = False
        for virtual_dart_id, virtual_dart in tracks["virtual_darts"].items():
            if part_id in virtual_dart["parts"].values():
                part_already_linked = True
                break

        # If part is not already linked, create a new virtual dart
        if not part_already_linked:
            virtual_dart_id = part_id
            tracks["virtual_darts"][virtual_dart_id] = {
                "bbox": part_bbox,
                "parts": {part_name: part_id, "tip": None, "barrel": None, "shaft": None, "flight": None},
                "frame_num": frame_num
            }
            # Add virtual dart to detected_darts
            detected_darts[virtual_dart_id] = tracks["virtual_darts"][virtual_dart_id]
            dart_data_copy = {key: value for key, value in detected_darts[virtual_dart_id].items() if key != 'frame_num'}
            tracks["linked_darts"][frame_num][virtual_dart_id] = dart_data_copy


        
        
    def find_closest_dart(self,part_bbox, detected_darts, frame_num, frame_window=60): #as a standard i put 60, because most of the videos are 30-60fps
        """
        Find the closest dart to the given part based on the bounding box center,
        considering a window of frames before and after the current frame.
        
        :param part: The part bounding box in the form [x1, y1, x2, y2] for the current frame.
        :param detected_darts: List of detected darts in the form of tuples [(bbox, frame_num), ...].
        :param frame_num: The frame number to filter the darts by.
        :param frame_window: The number of frames before and after to consider when matching darts to parts.
        
        :return: The closest dart's bounding box or None if no dart is found.
        """
    

        closest_dart = None
        closest_distance = float('inf')  # Initialize to a large value
        
        # Calculate the center of the part bounding box
        part_center_x, part_center_y = get_center_of_bbox(part_bbox)

        # Loop through darts in a window of frames (before and after the current frame)
        for dart_id, dart_data in detected_darts.items():  # Iterate over tracked darts
            
            dart_bbox = dart_data["bbox"]  # Extract bounding box
            dart_frame_num = dart_data["frame_num"]  # Extract frame number

            # Check if the dart is within the valid frame window
            if abs(dart_frame_num - frame_num) <= frame_window:
                dart_center_x, dart_center_y = get_center_of_bbox(dart_bbox)

                # Compute distance from part to dart center
                distance = math.sqrt((part_center_x - dart_center_x)**2 + (part_center_y - dart_center_y)**2)

                # Update the closest dart if it's the closest so far
                if distance < closest_distance:
                    closest_distance = distance
                    closest_dart = dart_id  # Store the dart ID instead of the bbox
                    
        return closest_dart
        
    def are_bboxes_close(self,bbox1, bbox2, threshold=50):
        """
        Check if two bounding boxes are close to each other based on their center distance.
        
        :param bbox1: The first bounding box as [x1, y1, x2, y2].
        :param bbox2: The second bounding box as [x1, y1, x2, y2].
        :param threshold: The maximum distance allowed between the centers of the two bounding boxes.
        
        :return: True if the bounding boxes are close, False otherwise.
        """
        
        # Calculate the center of the first bounding box
        center_x1 = (bbox1[0] + bbox1[2]) / 2
        center_y1 = (bbox1[1] + bbox1[3]) / 2
        
        # Calculate the center of the second bounding box
        center_x2 = (bbox2[0] + bbox2[2]) / 2
        center_y2 = (bbox2[1] + bbox2[3]) / 2
        
        # Calculate the Euclidean distance between the centers of the two bounding boxes
        distance = math.sqrt((center_x1 - center_x2)**2 + (center_y1 - center_y2)**2)
        
        # Check if the distance is less than or equal to the threshold
        return distance <= threshold
    
    
    
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            linked_darts_dic = {}  # Collect all darts for this frame
            board_dict = tracks["board"][frame_num]
            
            #Draw darts
            for track_id, dart_frames in tracks["linked_darts"].items():
                if frame_num in dart_frames:  
                    linked_darts_dic[track_id] = dart_frames[frame_num]


            for dart in linked_darts_dic.values():
                frame = draw_rectangle(frame, dart["bbox"],"Dart",(0,0,255))
                
                
            #Draw board
            for track_id, board in board_dict.items():
                frame = draw_rectangle(frame, board["bbox"],"Board",(0,255,0))

            output_video_frames.append(frame)

        return output_video_frames
