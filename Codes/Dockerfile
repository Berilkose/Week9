
FROM python:3.10-slim


RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY Application/requirements.txt .
COPY Application/Pose.py .
COPY Application/face.py .
COPY Application/TestData/ ./TestData/

RUN pip install --no-cache-dir -r requirements.txt


CMD python Pose.py && python face.py