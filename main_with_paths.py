import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

# Define a zone polygon for annotation purposes
ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

# Input and output video paths
INPUT_VIDEO_PATH = "C:\\PYTHON\\Object_Counting\\goat.mp4"
OUTPUT_VIDEO_PATH = "C:\\PYTHON\\Object_Counting\\output.mp4"

# Initialize video capture
def initialize_video_capture(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    return cap

# Load YOLO model
def load_yolo_model(model_path: str = "yolov8s.pt"):
    try:
        model = YOLO(model_path)
    except Exception as e:
        raise IOError(f"Error loading YOLO model: {e}")
    return model

# Main function
def main():
    # Initialize video capture
    cap = initialize_video_capture(INPUT_VIDEO_PATH)
    # Load YOLO model
    model = load_yolo_model()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer for saving output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    # Initialize BoxAnnotator for drawing bounding boxes
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    # Calculate zone polygon coordinates based on video resolution
    zone_polygon = (ZONE_POLYGON * np.array([frame_width, frame_height])).astype(int)
    # Initialize PolygonZone for defining a detection zone
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=(frame_width, frame_height))
    # Initialize PolygonZoneAnnotator for annotating the detection zone
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.green(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    # Main loop for processing video frames
    while True:
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            print("Finished processing the video")
            break

        # Perform object detection using YOLOv8
        result = model(frame, agnostic_nms=True)[0]
        # Convert YOLOv8 detections to Supervision Detections format
        detections = sv.Detections.from_yolov8(result)
        # Generate labels for each detection
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        # Annotate the frame with bounding boxes and labels
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        # Trigger the detection zone based on the detections
        zone.trigger(detections=detections)
        # Annotate the frame with the detection zone
        frame = zone_annotator.annotate(scene=frame)      
        
        # Write the annotated frame to the output video file
        out.write(frame)

        # Display the annotated frame (optional)
        cv2.imshow("YOLOv8 Live Detection", frame)

        # Exit the loop if 'Esc' key is pressed
        if (cv2.waitKey(30) == 27):
            break

    # Release the video capture, video writer, and close all OpenCV windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
