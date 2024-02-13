from django.db import models

class JudgeStatus(models.Model):
    idx = models.AutoField(db_column='IDX', primary_key=True, db_comment='인덱스')
    s3_path = models.CharField(db_column='S3_PATH', max_length=200, db_comment='s3 비디오 경로')
    emo_result = models.CharField(db_column='EMO_RESULT', max_length=60, db_comment='감정추론 결과')
    cdate = models.DateTimeField(db_column='CDATE', db_comment='생성일자')

    class Meta:
        managed = False
        db_table = 'Judge_status'