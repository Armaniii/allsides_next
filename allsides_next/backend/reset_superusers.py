from django.contrib.auth import get_user_model

# Get the user model
User = get_user_model()

# Delete all superusers
superusers = User.objects.filter(is_superuser=True)
count = superusers.count()
superusers.delete()
print(f"Deleted {count} existing superuser(s)")

# Create new admin superuser
admin = User.objects.create_superuser(
    username='admin',
    email='admin@example.com',
    password='admin'
)
print(f"Created new superuser: {admin.username}")
print("You can now login with username 'admin' and password 'admin'") 