"""
Django settings for g4emma_project project.

Generated by 'django-admin startproject' using Django 1.11.5.

For more information on this file, see
https://docs.djangoproject.com/en/1.11/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.11/ref/settings/
"""

import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.11/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
with open('/opt/emma/g4emma_site_key/secret_key.txt') as f:
    SECRET_KEY = f.read().strip() 

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

ALLOWED_HOSTS = ['emma.triumf.ca', '127.0.0.1']

# Secutiry settings addition for deployment
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
X_FRAME_OPTIONS = 'DENY'


# Application definition

INSTALLED_APPS = [
    'g4emma.apps.G4EmmaConfig',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'channels'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'g4emma_project.urls'

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

WSGI_APPLICATION = 'g4emma_project.wsgi.application'


# Database
# https://docs.djangoproject.com/en/1.11/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(os.environ['G4EMMA_DB_PATH'], 'db.sqlite3'),
    }
}


# Password validation
# https://docs.djangoproject.com/en/1.11/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/1.11/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'America/Vancouver'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.11/howto/static-files/

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'static/')

# Media files (other stuff, like the simulation output and input)
MEDIA_ROOT = os.environ['G4EMMA_MEDIA_DIR']
MEDIA_URL = '/media/'

DATA_DIRS = MEDIA_ROOT + os.environ['G4EMMA_DATA_DIR']


# Channels
# https://channels.readthedocs.io/en/latest/index.html
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "asgi_redis.RedisChannelLayer",
        "CONFIG": {
            "hosts": [("localhost", 6379)],
        },
        "ROUTING": "g4emma.routing.channel_routing",
    },
}



# Logging settings
LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters':
        {
            'simple':
            {
                'format': '[%(asctime)s] %(levelname)s %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }, # simple
            'verbose':
            {
                'format': '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }, # verbose
        }, # formatters
        'handlers': 
        {
            'dev_logfile': 
            {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': (os.environ['G4EMMA_LOG_PATH']+'/g4emma_debug.log'),
                'maxBytes': 1024*1024*50, # 50MB
                'backupCount': 2,
                'formatter': 'verbose'
            }, # dev_logfile
            'production_logfile':
            {
                'level': 'ERROR',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': (os.environ['G4EMMA_LOG_PATH']+'/g4emma_production.log'),
                'maxBytes': 1024*1024*50, # 50MB
                'backupCount': 2,
                'formatter': 'simple'
            }, # production_logfile
        }, # handlers
        'loggers':
        {
            'django':
            {
                'handlers': ['dev_logfile', 'production_logfile']
            }, # django
            'django.request':
            {
                'handlers': ['dev_logfile'] # this is more to disable the mail_admin handler than to actually use it
            } #django.request
        }, # loggers
} # LOGGING

