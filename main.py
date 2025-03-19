from utils import read_video, save_video, get_center_of_bbox
from trackers import TrackerDarts, TrackerBull
from scores_assigner import ScoresAssigner

#Remember, right now the best model to use it is by far model 3

def main():
    video_frames = read_video("inputs/DartsInput2.mp4") # read video
    
    tracker_darts = TrackerDarts('models/tracker_darts.pt')
    tracks_darts = tracker_darts.get_object_tracks(video_frames,read_from_stub=False,stub_path="stubs/track_stubs.pkl")
    tracks_darts["linked_darts"] = tracker_darts.interpolate_darts_positions(tracks_darts)
    
    tracker_bull = TrackerBull('models/center_tracker.pt')
    tracks_bull = tracker_bull.get_object_tracks(video_frames)
    tracks_bull = tracker_bull.interpolate_bull_positions(tracks_bull)
    '''
    #Assign sector board

    for frame_num, dart_track in enumerate(tracks_darts['linked_darts']):
        
        dartboard_center = tracker_bull.get_center_of_bull(tracks_darts, frame_num) #continue from here...
        scores_assigner = ScoresAssigner(dartboard_center)
        
        # Get the dart's bounding box for the current frame (this is the tip of the dart)
        dart_bbox = tracks_darts['linked_darts'][frame_num]['bbox']
        print(f"dart_bbox: {dart_bbox}")
        
        # Assign the dart to a sector using the ScoresAssigner class
        dart_center = get_center_of_bbox(dart_bbox)
        sector = scores_assigner.assign_sector(dart_center)
        
        # Optionally, update the tracks with the assigned sector and score for visualization
        score = scores_assigner.assign_score(sector)
        tracks_darts['linked_darts'][frame_num][1]['sector'] = sector
        tracks_darts['linked_darts'][frame_num][1]['score'] = score
    '''
    #print("tracks__bull: ", tracks_bull)
    # Assign sector board
    for frame_num in range(len(tracks_darts['linked_darts'])):
        
        center = tracker_bull.get_center_of_bull(tracks_bull, frame_num)
        if center is None:
            #print("ciao0")
            #We have to handle it in some way, due to the zoom and other stuff
            continue
        else:
            x_center, y_center = center
        scores_assigner = ScoresAssigner(x_center, y_center)

        # Get darts for the current frame (if any)
        linked_darts_dic = tracks_darts['linked_darts'][frame_num] if frame_num < len(tracks_darts['linked_darts']) else {}

        if linked_darts_dic:
            # Get the last detected dart (highest track_id)
            last_track_id = max(linked_darts_dic.keys())
            dart_bbox = linked_darts_dic[last_track_id]["bbox"]
            x_center, y_center = get_center_of_bbox(dart_bbox)
            dart_center = [x_center, y_center]
            
            if last_track_id in tracks_darts["virtual_darts"]: #if it is virtual, i have to project it
                print("VIRTUAL")

                board_bbox = tracker_darts.get_latest_board_center(tracks_darts,frame_num)
                if board_bbox is not None:
                    print("Not None")
                    dart_center = scores_assigner.project_part_on_board(dart_center,board_bbox)
            
            # Assign the dart to a sector
            sector = scores_assigner.assign_sector(dart_center)
            # Update tracks with assigned sector and score
            score = scores_assigner.assign_score(sector)
            tracks_darts['linked_darts'][frame_num][last_track_id]['sector'] = sector
            print("sector: ", sector)
            tracks_darts['linked_darts'][frame_num][last_track_id]['score'] = score
            print("score: ", score)
        
        '''    
        # Get the latest board bbox
        board_bbox = self.get_latest_board_center(tracks, frame_num)
        if board_bbox is None:
            return None  # No board detected, cannot project
        '''
     
    output_video_frames = tracker_darts.draw_annotations(video_frames,tracks_darts)
    output_video_frames = tracker_bull.draw_annotations(output_video_frames,tracks_bull)
    output_video_frames = scores_assigner.draw_scores(output_video_frames,tracks_darts)
    
    save_video(output_video_frames, "outputs/output_video_test4.avi")
    

    
if __name__ == "__main__":
    main()
    
    

'''
What is missing?
1. Writing the value of the dart in the frame
2. Do the rectangle of the dart bigger, just xn for some values
3. Fix the perspective
4. Add in the training bull_tracker

'''