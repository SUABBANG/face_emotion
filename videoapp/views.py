import os

from keras.models import load_model
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from django.http import JsonResponse
from django.conf import settings

from src.utils.datasets import get_labels
from src.utils.inference import load_detection_model

# loading models
detection_model_path = './model/haarcascade_frontalface_default.xml'
emotion_model_path = './model/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

face_detection = load_detection_model(detection_model_path)
emotion_model_path = load_model(emotion_model_path, compile=False)

class video_emotion(APIView):
    # post ~ 분석
    parser_classes = [MultiPartParser]
    def post(self, request, format=None, *args, **kwargs):
        if 'file' in request.FILES:
            uploaded_file = request.FILES['file']
            # 파일 로컬 저장
            local_file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
            print(local_file_path)
            with open(local_file_path, 'wb') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            return JsonResponse({"message": "success"})
        return JsonResponse({"message": "not Post"})

    # s3 upload
    # save database