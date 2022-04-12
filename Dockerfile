FROM python:3.8-buster
run pip install mediapipe
copy . /app

workdir /app

run pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

run python background_calculation.py
EXPOSE 8000
cmd ["python3", "app.py"]