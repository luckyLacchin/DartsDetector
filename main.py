from utils import read_video, save_video
from trackers_darts import TrackerDarts

#Remember, right now the best model to use it is by far model 3

def main():
    video_frames = read_video("inputs/DartsInput2.mp4") # read video
    
    tracker_darts = TrackerDarts('models/tracker_darts.pt')
    tracks_darts = tracker_darts.get_object_tracks(video_frames,read_from_stub=False,stub_path="stubs/track_stubs.pkl")
    
    interpolated_darts = tracker_darts.interpolate_darts_positions(tracks_darts)
    #print(f"Interpolated_darts: {interpolated_darts}")
    output_video_frames = tracker_darts.draw_annotations(video_frames,interpolated_darts)
    save_video(output_video_frames, "outputs/output_video.avi") # save video
    
    
    
if __name__ == "__main__":
    main()
    
    
#darts detection model it should be fine
#now i'm training the number detection model. If is not working, we are going to use the corners, i think (corners in: https://universe.roboflow.com/model-training-inclp/dart-blade-intersection/browse)
#then sectors training in: https://universe.roboflow.com/dataquartz/dartboard2/images/soPjs4xDk8jJhDXx8SwV?queryText=&pageSize=50&startingIndex=0&browseQuery=true
#then double, triples, training (rings training) https://universe.roboflow.com/dataquartz/dartboard2/images/soPjs4xDk8jJhDXx8SwV?queryText=&pageSize=50&startingIndex=0&browseQuery=true
#The interpolation has to be solved
#We have to associate the different parts of the dart to the same one...it would solve many problems
#now i have to interpolate them!
#i have to edit also the drawings...now i do just the interpolation