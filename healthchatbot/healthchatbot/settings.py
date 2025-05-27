import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
# Đảm bảo có đủ các ứng dụng được cài đặt
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'chatbot',
]

# Cấu hình middleware cần thiết
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# Cấu hình templates
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# Cấu hình REST Framework
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10
}
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True  # Đặt là True trong môi trường phát triển

# Nếu DEBUG là False, bạn phải thiết lập ALLOWED_HOSTS
ALLOWED_HOSTS = ['localhost', '127.0.0.1']
# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'
# Cấu hình URLs chính
ROOT_URLCONF = 'healthchatbot.urls'
# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-c8+9zp59x!u7k3yp*8f^$s=0xyy6lm4x8^=nb0b&kj7sj$l_n3'
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",      # React dev server
    "http://127.0.0.1:3000",      # React dev server alternative
    "http://localhost:8000",      # Django self
    "http://127.0.0.1:8000",      # Django self alternative
]

CORS_ALLOWED_ORIGIN_REGEXES = [
    r"^http://localhost:\d+$",    # Any localhost port
    r"^http://127\.0\.0\.1:\d+$", # Any 127.0.0.1 port
]

CORS_ALLOW_CREDENTIALS = True

CORS_ALLOW_ALL_ORIGINS = False  # Set to True only for development

# Additional CORS headers
CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]

CORS_ALLOW_METHODS = [
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]

# Allow all headers (for development)
CORS_ALLOW_ALL_HEADERS = True

# CSRF Settings
CSRF_TRUSTED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

ALLOWED_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0']