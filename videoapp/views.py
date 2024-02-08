from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from django.http import JsonResponse

class video_emotion(APIView):
    # post ~ 분석
    parser_classes = [MultiPartParser]
    def post(self, request, format=None, *args, **kwargs):
        if 'file' in request.FILES:
            uploaded_file = request.FILES['file']

            return JsonResponse({"message": "success"})
        return JsonResponse({"message": "not Post"})

    # s3 upload
    # save database