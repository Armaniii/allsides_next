# Generated by Django 4.2.18 on 2025-02-05 21:06

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0005_user_bias_rating_alter_user_email'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='allstars',
            field=models.IntegerField(default=0),
        ),
        migrations.CreateModel(
            name='ArgumentRating',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('stance', models.CharField(max_length=255)),
                ('core_argument', models.TextField()),
                ('rating', models.CharField(choices=[('L', 'Left'), ('LL', 'Lean Left'), ('C', 'Center'), ('LR', 'Lean Right'), ('R', 'Right')], max_length=2)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('query', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='argument_ratings', to='api.query')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='argument_ratings', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-created_at'],
                'unique_together': {('user', 'query', 'stance', 'core_argument')},
            },
        ),
    ]
