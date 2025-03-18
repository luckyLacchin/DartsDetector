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

            # Assign the dart to a sector
            x_center, y_center = get_center_of_bbox(dart_bbox)
            dart_center = [x_center, y_center]
            sector = scores_assigner.assign_sector(dart_center)
            # Update tracks with assigned sector and score
            score = scores_assigner.assign_score(sector)
            tracks_darts['linked_darts'][frame_num][last_track_id]['sector'] = sector
            tracks_darts['linked_darts'][frame_num][last_track_id]['score'] = score

    
     
    output_video_frames = tracker_darts.draw_annotations(video_frames,tracks_darts)
    output_video_frames = tracker_bull.draw_annotations(output_video_frames,tracks_bull)
    
    save_video(output_video_frames, "outputs/output_video_test3.avi")
    
    
    
if __name__ == "__main__":
    main()
    
    

'''
What are the next steps to do?
1. Solve absolute position even though there is zoom (it would be fondamental in order to get the points)
2. Training model for finding the center (inner and outer bullet) from that we can find the
points/value of the darts thrown
3. After got it, dysplay on the video the value of it
'''

'''
Next things to do:
- maybe redo the training of the first model used but for 400 trials, maybe a bit more...
- see how the bbox of the board is.
- use the tip and the angle in order to get the points
- we can detect also the board (confidence > 85) and the bullseye
- to handle the camera movements for the zoom and perspective for the angle


Tbh idk if it makes sense to interpolate the position of the board, due to zoom and also because the board is not always in the frame...To know the board, it would be very useful to do the projection of the dart on the board in order to get the points...I think that first, it would be very useful to redo the first model detect...https://universe.roboflow.com/julian-mowkv/detectdarts/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
For the perspective estimator i need a model for this: https://universe.roboflow.com/model-training-inclp/dart-blade-intersection/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true


This is how we are gonna detect points:
1. We are gonna use the bulleye. Given the position of it and of the tip, we are gonna measure the angle between the center and the tip. For the perspective problem, we are going to use the outer points in order to know the real angle.
2. When we don't have the tip, we are going to project the other parts of the dart on the board and then we are gonna do the same thing as before.
3. When there is the zoom and the board is not detected anymore, so we are interpolate it, until we don't detect it once more and so we are updating it
4. When the bulleye is not detected we are interpolating it until we don't update it

One day I have also to modify to usage also of the stub_path
TODO:
We have to try to interpolate both BOARD and BULL!!!
Then retrain the detect darts
Just use the angles, bull and the position of the darts in order to compute the points...we can't fix the perspective..

In the case that i'm drawing a part of the dart that is not the dart itself, i could do it bigger...ofc it is in the case of virtual_darts..

18 degrees per sector
TODO
Now it should work, i have to adjust the angle for the sector assignments!

'''