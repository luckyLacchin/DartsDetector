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

class TrackerDarts:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()
        self.last_interpolated_darts = []  # To keep track of the last 3 interpolated darts
        
    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.75) #before the confidence was 0.1, I consider it too low
            detections += detections_batch
            #print("ciao", detections)
            #break
        return detections
    
    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    position= get_center_of_bbox(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
    
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
                "linked_darts": [],
                "virtual_darts": {}  # Keep track of virtual darts
            }

            # Process each frame
            for frame_num, detection in enumerate(detections):
                cls_names = detection.names
                cls_names_inv = {v: k for k, v in cls_names.items()}

                detection_supervision = sv.Detections.from_ultralytics(detection)
                detections_with_tracks = self.tracker.update_with_detections(detection_supervision)

                tracks["darts"].append({})
                tracks["tips"].append({})
                tracks["barrels"].append({})
                tracks["shafts"].append({})
                tracks["flights"].append({})
                tracks["linked_darts"].append({})

                detected_darts = {}
                detected_parts = []

                # Process detections in each frame
                for frame_detection in detections_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]

                    if cls_id == cls_names_inv["Dart"]:
                        detected_darts[track_id] = {"bbox": bbox, "parts": {"tip": None, "barrel": None, "shaft": None, "flight": None}}
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

                # Step 1: Try to associate each part with the closest detected dart
                for part_name, part_id, part_bbox in detected_parts:
                    linked_dart_id = self.find_closest_dart(part_bbox, detected_darts, tracks["virtual_darts"], frame_num)

                    if linked_dart_id is not None:
                        # If this dart already has a different part of the same type, create a new virtual dart
                        if detected_darts[linked_dart_id]["parts"][part_name] is not None:
                            virtual_dart_id = f"virtual_{part_id}"
                            tracks["virtual_darts"][virtual_dart_id] = {
                                "bbox": part_bbox,
                                "parts": {part_name: part_id, "tip": None, "barrel": None, "shaft": None, "flight": None}
                            }
                            tracks["linked_darts"][frame_num][virtual_dart_id] = tracks["virtual_darts"][virtual_dart_id]
                        else:
                            detected_darts[linked_dart_id]["parts"][part_name] = part_id
                            tracks["linked_darts"][frame_num][linked_dart_id] = detected_darts[linked_dart_id]

                # Step 2: If a part has no dart, assign it to a new virtual dart
                for part_name, part_id, part_bbox in detected_parts:
                    if not any(part_id in detected_darts[dart_id]["parts"].values() for dart_id in detected_darts):
                        virtual_dart_id = f"virtual_{part_id}"
                        tracks["virtual_darts"][virtual_dart_id] = {
                            "bbox": part_bbox,
                            "parts": {"tip": None, "barrel": None, "shaft": None, "flight": None}
                        }
                        tracks["virtual_darts"][virtual_dart_id]["parts"][part_name] = part_id
                        tracks["linked_darts"][frame_num][virtual_dart_id] = tracks["virtual_darts"][virtual_dart_id]

                # Step 3: Merge virtual darts with newly detected darts
                for dart_id, dart_data in detected_darts.items():
                    for virtual_dart_id, virtual_dart_data in list(tracks["virtual_darts"].items()):
                        if self.are_bboxes_close(dart_data["bbox"], virtual_dart_data["bbox"]):
                            # Merge virtual dart with real dart
                            for part_name, part_id in virtual_dart_data["parts"].items():
                                if part_id is not None and dart_data["parts"][part_name] is None:
                                    dart_data["parts"][part_name] = part_id

                            # Remove virtual dart since it's now linked to a real one
                            del tracks["virtual_darts"][virtual_dart_id]
                """"
                # If we already have 3 darts, stop interpolating until new darts are thrown
                if dart_count < 3:
                    # Interpolate only the last 3 darts
                    last_interpolated_darts = self.last_interpolated_darts[-3:]
                    if dart_count == 0:
                        self.last_interpolated_darts.clear()  # Clear if no darts yet (first frame)

                    # Interpolate new darts based on existing ones
                    new_tip_positions = {track_id: tracks["tips"][frame_num][track_id] for track_id in last_interpolated_darts}
                    tracks["tips"][frame_num] = self.interpolate_tip_positions(new_tip_positions)
                    self.last_interpolated_darts.extend([track_id for track_id in tracks["tips"][frame_num].keys() if track_id not in self.last_interpolated_darts])
                    this part is to be corrected
                """
            if stub_path is not None:
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracks, f)

            return tracks
        
        
    def find_closest_dart(self, part_bbox, detected_darts):
        """
        find the closest dart (or create a virtual one) for a given part.
        """
        if not detected_darts:
            return None  #no darts detected in this frame

        part_center = np.array([(part_bbox[0] + part_bbox[2]) / 2, (part_bbox[1] + part_bbox[3]) / 2])
        
        min_distance = float("inf")
        closest_dart_id = None
        
        for dart_id, dart_data in detected_darts.items():
            dart_bbox = dart_data["bbox"]
            dart_center = np.array([(dart_bbox[0] + dart_bbox[2]) / 2, (dart_bbox[1] + dart_bbox[3]) / 2])
            
            distance = np.linalg.norm(part_center - dart_center)
            
            if distance < min_distance:
                min_distance = distance
                closest_dart_id = dart_id

        return closest_dart_id
    
    
            
            
         
    def draw_annotations(self,video_frames, tracks):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            #print ("frame_num: ", frame_num)
            #print(f"Total Frames in Tracks: {len(tracks['Players'])}")

            tip_dic = tracks["darts"][frame_num]

            # Draw Tips
            for track_id, tip in tip_dic.items():
                frame = self.draw_ellipse(frame, tip["bbox"],track_id)
            
            output_video_frames.append(frame)

        return output_video_frames
    
    
    def draw_ellipse(self,frame,bbox,track_id=None):
        y2 = int(bbox[3]) #bottom of the bounding box
        x_center, _ = get_center_of_bbox(bbox) #center in the middle of x
        width = get_bbox_width(bbox)
        smaller_factor = 0.5
        
        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width * smaller_factor), int(0.35*width*smaller_factor)),
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
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          (255, 255, 0),
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
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
    
    
    def draw_triangle(self,frame,bbox): #in reality in the future i could also do a circle around the ball, it would look better probably. Anyway the ball doesn't work so much, because i should train more the model
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,(255, 255, 0), cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame
    
    
    def draw_circle(self, frame, bbox):
        x_center = (int)((bbox[0] + bbox[2])) // 2
        y_center = (int)((bbox[1] + bbox[3])) // 2

        radius = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.6)  # 0.6 instead of 0.5

        cv2.circle(frame, (x_center, y_center), radius, (255, 255, 0), thickness=2)

        return frame
    
    
    
    
    '''
                # Step 3: Merge virtual darts with real darts when close enough
            for dart_id, dart_data in detected_darts.items():
                for virtual_dart_id, virtual_dart_data in list(tracks["virtual_darts"].items()):
                    if self.are_bboxes_close(dart_data["bbox"], virtual_dart_data["bbox"]):
                        # Merge virtual dart with real dart
                        for part_name, part_id in virtual_dart_data["parts"].items():
                            if part_id is not None and dart_data["parts"][part_name] is None:
                                dart_data["parts"][part_name] = part_id

                        # Remove virtual dart since it's now linked to a real one
                        del tracks["virtual_darts"][virtual_dart_id]
    
    
    
    def remove_virtual_dart_if_detected(self, detected_darts, tracks, frame_num, part_name, part_id, part_bbox):
        # Check if the part corresponds to a virtual dart
        virtual_dart_id = None
        for virtual_dart_id, virtual_dart in tracks["virtual_darts"].items():
            if virtual_dart.get(part_name) == part_id:  # If the part matches
                virtual_dart_id = virtual_dart_id
                break

        if virtual_dart_id is not None:
            # Virtual dart found, check if all parts are now detected
            virtual_dart = tracks["virtual_darts"][virtual_dart_id]
            
            # Check if all parts of the dart are now detected and complete
            if all(virtual_dart.get(part) is not None for part in ["tip", "barrel", "shaft", "flight"]):
                # All components are linked to the dart, so we can delete the virtual dart
                del tracks["virtual_darts"][virtual_dart_id]
                print(f"Removed virtual dart {virtual_dart_id} as a real dart was detected.")
                
                # Link the real dart to tracks["linked_darts"]
                detected_darts[virtual_dart_id]["frame_num"] = frame_num
                tracks["linked_darts"][frame_num][virtual_dart_id] = detected_darts[virtual_dart_id]
            else:
                print(f"Virtual dart {virtual_dart_id} is not yet complete.")
                
                
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
        
        '''