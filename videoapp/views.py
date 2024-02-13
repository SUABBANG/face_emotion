import os

from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from django.http import JsonResponse
from django.conf import settings
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