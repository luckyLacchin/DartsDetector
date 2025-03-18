import math
from utils import get_center_of_bbox

class ScoresAssigner:
    def __init__(self, x_center, y_center, sectors=20):
        # dartboard_center is the center of the bullseye (x, y)
        self.dartboard_center = [x_center, y_center]
        self.sectors = sectors


    def calculate_angle(self, dart_center):
        """
        Calculate the angle between the dart and the center of the bullseye.
        """
        dx = dart_center[0] - self.dartboard_center[0]
        dy = dart_center[1] - self.dartboard_center[1]
        
        # Calculate the angle in radians
        angle = math.atan2(dy, dx)  # Angle from the horizontal axis (in radians)
        
        # Convert angle to degrees (from 0 to 360 degrees)
        angle_deg = math.degrees(angle) % 360  # Ensuring it's positive
        
        return angle_deg

    def assign_sector(self, dart_center):
        """
        Assign the sector based on the angle.
        Each sector is 18 degrees wide (360 / 20 = 18).
        """
        angle = self.calculate_angle(dart_center)
        
        # Determine the sector number (0-19)
        sector_number = int(angle // (360 / self.sectors))  # Integer division
        return sector_number

    def assign_score(self, sector):
        """
        Return the score based on the dartboard sector.
        This can be customized based on dartboard layout.
        """
        # This has to be edited maybe
        sector_scores = {
            0: 20, 1: 1, 2: 18, 3: 4, 4: 13, 5: 6, 6: 10, 7: 15, 8: 2, 9: 17,
            10: 3, 11: 19, 12: 7, 13: 16, 14: 8, 15: 11, 16: 9, 17: 14, 18: 12, 19: 5
        }
        return sector_scores.get(sector, 0)

    def link_dart_to_sector(self, part_bbox, detected_darts, frame_num, tracks, part_name=None):
        """
        Link the dart to the appropriate sector and assign a score.
        If part_name is not None, it represents another part besides the tip (used for projecting parts).
        """
        dart_center = get_center_of_bbox(part_bbox)
        
        # If it's the dart tip (primary part), use it directly to calculate the sector
        if part_name is None or part_name == 'tip':
            sector = self.assign_sector(dart_center)
            score = self.assign_score(sector)
        else:
            # For other parts (e.g., wing), project them onto the dartboard
            # We'll assume here that we should adjust the position based on the part's relative position.
            projected_center = self.project_part_on_board(dart_center)
            sector = self.assign_sector(projected_center)
            score = self.assign_score(sector)

'''
# Usage example:

# Assume we have the center of the bullseye
dartboard_center = (300, 300)  # Example center coordinates (can be changed)

# Example part_bbox of a detected dart (this is the dart tip)
part_bbox = (280, 250, 320, 290)

# Initialize the ScoresAssigner
scores_assigner = ScoresAssigner(dartboard_center)

# Detected darts and tracks (for example purposes, these should be filled with actual data)
detected_darts = {}
tracks = {"linked_darts": {}}

# Link the dart to a sector and assign score (using the dart tip's position)
frame_num = 1
scores_assigner.link_dart_to_sector(part_bbox, detected_darts, frame_num, tracks)

# Print the updated tracks to check the results
print(tracks)
'''
        