import os

APP_NAME = 'VoiceChanger'
DATA_DIR_PATH = os.path.join(os.getenv('LOCALAPPDATA'), APP_NAME)
PTH_DIR_PATH = os.path.join(DATA_DIR_PATH, 'pth')
INDEX_DIR_PATH = os.path.join(DATA_DIR_PATH, 'index')
SETTING_FILE_PATH = os.path.join(DATA_DIR_PATH, 'settings.json')
SPEAKERS_FILE_PATH = os.path.join(DATA_DIR_PATH, 'speakers.json')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
