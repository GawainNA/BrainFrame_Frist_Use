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
        capsule_names = ["object_detector"]
    detections = api.process_image(frame, capsule_names, {})
    return detections
 
def detect_person(detections, frame, output_path):
    # label face
    for detected_obj in detections:
        if detected_obj.class_name == 'face':
            coords = detected_obj.coords
            cv2.rectangle(frame, (coords[0][0], coords[0][1]), (coords[2][0], coords[2][1]), (0,0,255),1)

    # crop person and output
    file_index = 0
    for detected_obj in detections:
        if detected_obj.class_name == 'person':
            file_index+=1
            coords = detected_obj.coords
            cropped = frame[coords[0][1]:coords[2][1], coords[0][0]:coords[2][0]]
            cv2.imwrite(output_path+str(file_index)+'.png',cropped)
    
    print("Output successfully")

def main():
    # The capsules for person and face detection
    capsule_names = ["object_detector"]
 
    # The video file name, it can be replaced by the other video file or rtsp/http streams
    stream_path = "./videos/201202_01_Oxford Shoppers_4k_048.mp4"

    # The output images
    output_path = './output/person/'


    # The url to access the brainframe server with rest api
    bf_server_url = "http://localhost"
 
    api = BrainFrameAPI(bf_server_url)
    api.wait_for_server_initialization()
 
    frame = read_frame(stream_path, 5)
    if frame is None:
        return
 
    detections = detect_image(api, frame, capsule_names)

    print(detections)

    # crop persons as images
    #detect_person(detections, frame, output_path)

if __name__ == "__main__":
    main()