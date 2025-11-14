import cv2
import os
import time
import numpy as np

import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

class FaceDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.isReady = self.prepare_camera_and_window()

        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = self.FaceLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            result_callback=self.print_result)
        
        self.latest_result = None
        
    def print_result(self, result, output_image, timestamp_ms):
        # print('face landmarker result: {}'.format(result))
        self.latest_result = result

    def prepare_camera_and_window(self) -> bool:
        cv2.namedWindow("preview")
        self.videoCapture = cv2.VideoCapture(0)

        if self.videoCapture.isOpened(): 
            rval, frame = self.videoCapture.read()
        else:
            rval = False

        return rval
    
    def start(self):

        if not self.isReady:
            print("Camera not ready...")
            return

        with self.FaceLandmarker.create_from_options(self.options) as landmarker:

            run_stream = True
            while run_stream:
                run_stream, frame = self.videoCapture.read()
                frame_timestamp_ms = int(time.time() * 1000)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                landmarker.detect_async(mp_image, frame_timestamp_ms)
                blank = np.zeros_like(frame)
                
                # annotated = self.draw_landmarks_on_image(frame, self.latest_result)
                annotated = self.draw_landmarks_on_image(blank, self.latest_result)
                

                cv2.imshow("preview", annotated)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    run_stream = False
                    # break


    def draw_landmarks_on_image(self,rgb_image, detection_result):

        if detection_result is None or detection_result.face_landmarks is None:
            return rgb_image  # return original frame
        
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]
    

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())

        return annotated_image

        

# def capture_webcam():
#     cv2.namedWindow("preview")
#     videoCapture = cv2.VideoCapture(0)

#     if videoCapture.isOpened():
#         rval, frame = videoCapture.read()
#     else:
#         rval = False

#     while rval:
#         cv2.imshow("preview", frame)
#         rval, frame = videoCapture.read()
        
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)



#         key = cv2.waitKey(20)
#         if key == 27:
#             rval = False

#     cv2.destroyWindow("preview")
#     videoCapture.release()


if __name__ == "__main__":

    model_path = 'face_landmarker.task'


    fd = FaceDetector(model_path)

    fd.start()



