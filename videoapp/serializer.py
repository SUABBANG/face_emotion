from django.db import models


class Judge_status(models.Model):
    IDX = models.AutoField(primary_key=True)
    S3_PATH = models.CharField(max_length=200)
    EMO_RESULT = models.CharField(max_length=60)
    CDATE = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = False
        db_table = 'Judge_status'