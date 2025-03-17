import cv2
from utils import get_bbox_width, get_center_of_bbox

def draw_rectangle(frame, bbox, label_text, color=(255, 255, 0)):
    
    """
    Draws a bounding box around the dart with a label.
        
    :param frame: The image frame.
    :param bbox: Bounding box coordinates (x1, y1, x2, y2).
    """
    if bbox is None or len(bbox) != 4:
        print("Invalid bounding box:", bbox)
        return frame  # Avoid crashing if bbox is not valid

    # Convert to integers to avoid OpenCV errors
    x1, y1, x2, y2 = map(int, bbox)

    # Draw main bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=3)

    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

    # Draw background rectangle for text
    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, cv2.FILLED)

    # Put label text
    cv2.putText(frame, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return frame

       

    
def draw_ellipse(frame,bbox,track_id=None,color=(255, 255, 0)):
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
    
