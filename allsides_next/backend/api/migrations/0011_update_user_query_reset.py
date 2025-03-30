from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('api', '0010_add_stance_to_thumbsrating'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='last_query_reset',
            field=models.DateTimeField(null=True, blank=True),
        ),
        migrations.RemoveField(
            model_name='user',
            name='reset_time',
        ),
    ] 