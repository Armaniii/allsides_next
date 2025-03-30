from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('api', '0009_thumbsrating'),
    ]

    operations = [
        migrations.AddField(
            model_name='thumbsrating',
            name='stance',
            field=models.CharField(max_length=255, default=''),
            preserve_default=False,
        ),
    ] 