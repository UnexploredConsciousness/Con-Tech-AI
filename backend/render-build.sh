#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Installing FFmpeg directly (static build)..."
# Render native Python environments do not have apt-get sudo privileges.
# We download a static build of FFmpeg and place it in the path to support librosa/video processing.
mkdir -p /opt/render/project/bin
curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar xJC /tmp
cp /tmp/ffmpeg-*-amd64-static/ffmpeg /opt/render/project/bin/
cp /tmp/ffmpeg-*-amd64-static/ffprobe /opt/render/project/bin/
chmod +x /opt/render/project/bin/ffmpeg
chmod +x /opt/render/project/bin/ffprobe

echo "Downloading ML models to cache during build to prevent cold-start out-of-memory/timeouts..."
python download_models.py

echo "Build complete."
