import cv2
from brainframe.api import BrainFrameAPI
 
 
def read_frame(stream_uri, frame_index):
    cap = cv2.VideoCapture(stream_uri)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    rst, frame = cap.read()
    if not rst:
        print(f"Failed to read frame: {frame_index}")
    cap.release()
    return frame
 
 
def detect_image(api, frame, capsule_names=None):
    if capsule_names is None:
        capsule_names = ["detector_person_openvino", "detector_face_openvino"]
    detections = api.process_image(frame, capsule_names, {})
    return detections
 

def detect_person_vehicle(video_path, output_video_path, api, capsule_names):
    capture = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path ,fourcc, 30.0, (1280,720), True)
    print("Start processing.....")
    if capture.isOpened():
        while True:
            ret,frame = capture.read()
            if not ret:break
            detections = detect_image(api, frame, capsule_names)
            for detected_obj in detections:
                if detected_obj.class_name == 'person':
                    coords = detected_obj.coords
                    cv2.rectangle(frame, (coords[0][0], coords[0][1]), (coords[2][0], coords[2][1]), (0,0,255),1)
                elif detected_obj.class_name == 'vehicle':
                    coords = detected_obj.coords
                    cv2.rectangle(frame, (coords[0][0], coords[0][1]), (coords[2][0], coords[2][1]), (255,0,0),1)
            writer.write(frame)
    else:
        print('Failed to open video.')
    writer.release()
    print("Process successfully!")

def main():
    # The capsules for person and vehicle detecion
    capsule_person_vehicle_names = ["detector_person_vehicle_bike_openvino"]
 
    # The video file name, it can be replaced by the other video file or rtsp/http streams
    stream_path = "./videos/London_walk.mp4"

    # The output video
    output_video_path = './output/person_vehicle/detected.mp4'
 
    # The url to access the brainframe server with rest api
    bf_server_url = "http://localhost"
 
    api = BrainFrameAPI(bf_server_url)
    api.wait_for_server_initialization()
    
    # Detect persons and vehicles, output as video 
    detect_person_vehicle(stream_path, output_video_path, api, capsule_person_vehicle_names)
if __name__ == "__main__":
    main()