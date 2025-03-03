from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
sys.path.append("../")
from utils import get_center_of_bbox, get_bbox_width
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
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.6) #before the confidence was 0.1, I consider it too low
            detections += detections_batch
            #print("ciao", detections)
            #break
        return detections
    
    def interpolate_tip_positions(self, tip_positions):
        """
        Interpolate tip positions until there are 3 darts on the board, keeping track_id.
        Only interpolate the last 3 darts thrown.
        """
        # Prepare data for interpolation: each entry should include track_id and bbox (x1, y1, x2, y2)
        interpolated_data = []

        # For each track_id in tip_positions, extract track_id and its corresponding bbox (x1, y1, x2, y2)
        for track_id, data in tip_positions.items():
            if "bbox" in data:
                interpolated_data.append({"track_id": track_id, **data})

        # Convert to DataFrame for interpolation (track_id, x1, y1, x2, y2)
        df_tip_positions = pd.DataFrame(interpolated_data, columns=["track_id", "x1", "y1", "x2", "y2"])

        # If there are less than 3 darts, interpolate missing positions
        if len(df_tip_positions) < 3:
            df_tip_positions = df_tip_positions.interpolate()  # Interpolate missing positions
            df_tip_positions = df_tip_positions.bfill()  # Backfill ensures any remaining missing values at the beginning are filled

        # Recreate the tip positions with track_id and interpolated bbox
        interpolated_tip_positions = {}
        for _, row in df_tip_positions.iterrows():
            interpolated_tip_positions[row["track_id"]] = {
                "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]]
            }

        return interpolated_tip_positions

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

            tracks = {
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

                detected_parts = []

                # Step 1: Detect parts and link to darts (if possible)
                for frame_detection in detections_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]

                    if cls_id == cls_names_inv["Dart"]:
                        detected_darts[track_id] = {"bbox": bbox, "parts": {"tip": None, "barrel": None, "shaft": None, "flight": None}, "frame_num": frame_num}
                        tracks["darts"][frame_num][track_id] = {"bbox": bbox}

                    elif cls_id == cls_names_inv["Tip"]:
                        detected_parts.append(("tip", track_id, bbox))
                        tracks["tips"][frame_num][track_id] = {"bbox": bbox}

                    elif cls_id == cls_names_inv["Barrel"]:
                        detected_parts.append(("barrel", track_id, bbox))
                        tracks["barrels"][frame_num][track_id] = {"bbox": bbox}

                    elif cls_id == cls_names_inv["Shaft"]:
                        detected_parts.append(("shaft", track_id, bbox))
                        tracks["shafts"][frame_num][track_id] = {"bbox": bbox}

                    elif cls_id == cls_names_inv["Flight"]:
                        detected_parts.append(("flight", track_id, bbox))
                        tracks["flights"][frame_num][track_id] = {"bbox": bbox}

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

            #print("tracks[linked_darts]", tracks["linked_darts"])
            for frame_num, linked_darts_data in enumerate(tracks["linked_darts"]):
                print(f"Starting linked_darts for frame {frame_num}:")
                print(linked_darts_data)
                if frame_num == 60:
                    break
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
            virtual_dart_id = f"virtual_{part_id}"
            tracks["virtual_darts"][virtual_dart_id] = {
                "bbox": part_bbox,
                "parts": {part_name: part_id, "tip": None, "barrel": None, "shaft": None, "flight": None},
                "frame_num": frame_num
            }
            # Add virtual dart to detected_darts
            detected_darts[virtual_dart_id] = tracks["virtual_darts"][virtual_dart_id]
            dart_data_copy = {key: value for key, value in detected_darts[virtual_dart_id].items() if key != 'frame_num'}
            tracks["linked_darts"][frame_num][virtual_dart_id] = dart_data_copy


        
        
    def find_closest_dart(self,part_bbox, detected_darts, frame_num, frame_window=60): #as a standard i put 60, because most of the videos are 30760fps
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
            
            linked_darts_dic = tracks["linked_darts"][frame_num]

            for track_id, dart in linked_darts_dic.items():
                
                frame = self.draw_rectangle(frame, dart["bbox"])

            output_video_frames.append(frame)

        return output_video_frames

    
    
    def draw_ellipse(self,frame,bbox,track_id=None):
        y2 = int(bbox[3]) #bottom of the bounding box
        x_center, _ = get_center_of_bbox(bbox) #center in the middle of x
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = (255, 255, 0),
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect)),
                          (int(x2_rect),int(y2_rect)),
                          (255, 255, 0),
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame
    
    
    def draw_rectangle(self, frame, bbox):
        x1, y1, x2, y2 = bbox

        cv2.rectangle(frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 255, 0),
                    thickness=2)
        
        cv2.rectangle(frame,
                    (int(x1)+10, int(y1)+20),
                    (int(x2), int(y2)),
                    (211, 211, 211),
                    cv2.FILLED)
        cv2.putText(frame,
                    "Dart",
                    (int(x1) + 10, int(y1) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2)

        return frame