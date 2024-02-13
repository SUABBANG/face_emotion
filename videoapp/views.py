import os
import cv2
import numpy as np
import heapq
import boto3

from keras.models import load_model
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from django.http import JsonResponse
from django.conf import settings
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from statistics import mode

from face_emotion_project.settings import env
from src.utils.datasets import get_labels
from src.utils.inference import detect_faces
from src.utils.inference import draw_text
from src.utils.inference import draw_bounding_box
from src.utils.inference import apply_offsets
from src.utils.inference import load_detection_model
from src.utils.preprocessor import preprocess_input
from videoapp.models import JudgeStatus

# loading models
detection_model_path = './model/haarcascade_frontalface_default.xml'
emotion_model_path = './model/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

from config import config
# AWS
BUCKET_NAME = config.aws_info['bucket_name']
AWS_ACCESS_KEY_ID = config.aws_info['aws_access_key_id']
AWS_SECRET_ACCESS_KEY = config.aws_info['aws_secret_access_key']
REGION = config.aws_info['region']
S3_VIDEO_PATH = 'ir_data/result_video'
DATE_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

class GetS3FileList:
    def __init__(self, prefix, aws_access_key_id, aws_secret_access_key, region, bucket_name, s3_video_path):
        self.client = boto3.client('s3',
                                   aws_access_key_id=aws_access_key_id,
                                   aws_secret_access_key=aws_secret_access_key,
                                   region_name=region)
        self.prefix = prefix
        self.bucket_name = bucket_name
        self.s3_video_path = s3_video_path

    def upload_mp4_file(self, file_name):
        try:
            local_file_path = f'{self.prefix}/{file_name}'
            s3_upload_key = f'{self.s3_video_path}/{file_name}'

            print(local_file_path)
            print(s3_upload_key)

            # S3 ���� ���ε� (�ۺ��� �б� ���� ����)
            self.client.upload_file(local_file_path, self.bucket_name, s3_upload_key,
                                    ExtraArgs={'ACL': 'public-read', 'ContentType': 'video/mp4'})

            return s3_upload_key
        except Exception as e:
            print(e)
            return False


class video_emotion(APIView):
    # post
    parser_classes = [MultiPartParser]
    def post(self, request, format=None, *args, **kwargs):
        if 'file' in request.FILES:
            uploaded_file = request.FILES['file']
            # 파일 로컬 저장
            local_file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
            with open(local_file_path, 'wb') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)
            print("LOCAL SAVE")
            # hyper-parameters for bounding boxes shape

            frame_window = 15
            emotion_offsets = (20, 40)
            emotion_target_size = emotion_classifier.input_shape[1:3]
            emotion_window = []

            video_capture = cv2.VideoCapture(local_file_path)

            fps = video_capture.get(cv2.CAP_PROP_FPS)
            width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            VIDEO_NAME = datetime.now().strftime("%f") + uploaded_file.name
            local_file_path_before = os.path.join(settings.MEDIA_ROOT, VIDEO_NAME)
            output_video = cv2.VideoWriter(local_file_path_before, fourcc, fps, (width, height))

            emotion_predictions = []

            while True:
                try:
                    bgr_image = video_capture.read()[1]
                    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
                    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                    faces = detect_faces(face_detection, gray_image)

                    for face_coordinates in tqdm(faces, desc='Processing faces', unit='faces'):
                        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                        gray_face = gray_image[y1:y2, x1:x2]
                        try:
                            gray_face = cv2.resize(gray_face, (emotion_target_size))
                        except Exception as e:
                            print("error : ",e)
                            continue

                        gray_face = preprocess_input(gray_face, True)
                        gray_face = np.expand_dims(gray_face, 0)
                        gray_face = np.expand_dims(gray_face, -1)
                        emotion_prediction = emotion_classifier.predict(gray_face)
                        emotion_probability = np.max(emotion_prediction)
                        emotion_label_arg = np.argmax(emotion_prediction)
                        emotion_text = emotion_labels[emotion_label_arg]
                        emotion_window.append(emotion_text)

                        if len(emotion_window) > frame_window:
                            emotion_window.pop(0)
                        try:
                            emotion_mode = mode(emotion_window)
                        except Exception as e:
                            continue
                        emotion_predictions.append([emotion_text, round(emotion_probability, 2)])

                        if emotion_text == 'angry':
                            color = emotion_probability * np.asarray((255, 0, 0))
                        elif emotion_text == 'sad':
                            color = emotion_probability * np.asarray((0, 0, 255))
                        elif emotion_text == 'happy':
                            color = emotion_probability * np.asarray((255, 255, 0))
                        elif emotion_text == 'surprise':
                            color = emotion_probability * np.asarray((0, 255, 255))
                        else:
                            color = emotion_probability * np.asarray((0, 255, 0))

                        color = color.astype(int)
                        color = color.tolist()

                        draw_bounding_box(face_coordinates, rgb_image, color)
                        draw_text(face_coordinates, rgb_image, emotion_mode,
                                  color, 0, -45, 1, 1)

                    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    output_video.write(bgr_image)
                    # cv2.imshow('window_frame', bgr_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    break

            video_capture.release()
            output_video.release()

            video_trans_name = 'trans_' + VIDEO_NAME
            video_trans_path = os.path.join(settings.MEDIA_ROOT, video_trans_name)

            os.system(f'C:/ffmpeg/bin/ffmpeg -i {local_file_path_before} -vcodec libx264 {video_trans_path}')

            emotion_counts = defaultdict(int)

            for emotion, _ in emotion_predictions:
                emotion_counts[emotion] += 1
            # Find the two most common emotions and their counts
            most_common_emotions = heapq.nlargest(6, emotion_counts.items(), key=lambda x: x[1])

            emo_result = {}

            if len(most_common_emotions) > 1:
                # Unpack the results
                first_emotion, fir_count = most_common_emotions[0]
                second_emotion, sec_count = most_common_emotions[1]
                all_datas = 0
                for i in range(len(most_common_emotions)):
                    all_datas += most_common_emotions[i][1]
                fir_proba = round(fir_count / all_datas * 100, 2)
                sec_proba = round(sec_count / all_datas * 100, 2)

                emo_result[first_emotion] = fir_proba
                emo_result[second_emotion] = sec_proba
            else:
                first_emotion, fir_count = most_common_emotions[0]
                fir_proba = 100
                emo_result[first_emotion] = fir_proba

            s3_uploader = GetS3FileList(prefix='./media/', aws_access_key_id=AWS_ACCESS_KEY_ID,
                                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region=REGION,
                                        bucket_name=BUCKET_NAME, s3_video_path=S3_VIDEO_PATH)
            path_result = s3_uploader.upload_mp4_file(file_name=video_trans_name)
            video_url = f"https://s3.{REGION}.amazonaws.com/{BUCKET_NAME}/{path_result}"

            input_data = [(video_url, str(emo_result), DATE_TIME)]

            print(input_data)

            for video_url, emo_result, date_time in input_data:
                JudgeStatus.objects.create(s3_path=video_url, emo_result=emo_result, cdate=date_time)

            return JsonResponse({"message": emo_result})
        return JsonResponse({"message": "not Post"})

