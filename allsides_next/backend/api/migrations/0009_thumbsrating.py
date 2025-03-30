from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0008_alter_argumentrating_unique_together_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='ThumbsRating',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('core_argument', models.TextField()),
                ('rating', models.CharField(choices=[('UP', 'Thumbs Up'), ('DOWN', 'Thumbs Down')], max_length=4)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('query', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='thumbs_ratings', to='api.query')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='thumbs_ratings', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
        migrations.AddConstraint(
            model_name='thumbsrating',
            constraint=models.UniqueConstraint(
                fields=('user', 'core_argument'),
                name='unique_user_argument_thumbs_rating'
            ),
        ),
    ] 