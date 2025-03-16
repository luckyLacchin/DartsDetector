from utils import read_video, save_video
from trackers import TrackerDarts

#Remember, right now the best model to use it is by far model 3

def main():
    video_frames = read_video("inputs/DartsInput2.mp4") # read video
    
    tracker_darts = TrackerDarts('models/tracker_darts.pt')
    tracks_darts = tracker_darts.get_object_tracks(video_frames,read_from_stub=False,stub_path="stubs/track_stubs.pkl")
    
    tracks_darts["linked_darts"] = tracker_darts.interpolate_darts_positions(tracks_darts)
     
    output_video_frames = tracker_darts.draw_annotations(video_frames,tracks_darts)
    save_video(output_video_frames, "outputs/output_video.avi") # save video
    
    
    
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

'''