import os
import re
from typing import Dict
import yt_dlp
from fastapi import APIRouter
from pydantic import BaseModel
from config import VIDEO_DIR

router = APIRouter()

class YouTubeRequest(BaseModel):
    url: str

# Dictionary to store download progress for each video ID
download_progress: Dict[str, Dict] = {}

def extract_youtube_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def progress_hook(d: Dict) -> None:
    """Hook to capture yt-dlp download progress."""
    video_id = d.get('info_dict', {}).get('id')
    if not video_id:
        return
    if d['status'] == 'downloading':
        downloaded_bytes = d.get('downloaded_bytes', 0)
        total_bytes = d.get('total_bytes', d.get('total_bytes_estimate', 0))
        speed = d.get('speed', 0) or 0
        eta = d.get('eta', 0) or 0
        progress = (downloaded_bytes / total_bytes) * 100 if total_bytes > 0 else 0
        download_progress[video_id] = {
            'progress': progress,
            'speed': speed / 1024 / 1024,  # Convert to MB/s
            'eta': eta,  # Seconds
            'status': 'downloading'
        }
    elif d['status'] == 'finished':
        download_progress[video_id] = {
            'progress': 100,
            'speed': 0,
            'eta': 0,
            'status': 'finished'
        }
    elif d['status'] == 'error':
        download_progress[video_id] = {
            'progress': 0,
            'speed': 0,
            'eta': 0,
            'status': 'error'
        }

@router.post("/load_youtube")
async def load_youtube(req: YouTubeRequest):
    video_id = extract_youtube_id(req.url)
    if not video_id:
        return {"status": "error", "message": "Invalid YouTube URL"}

    # Check if video exists in uploads folder
    for filename in os.listdir(VIDEO_DIR):
        if video_id in filename and filename.endswith('.mp4'):
            return {"status": "ok", "filename": filename}

    # Download video-only stream if it doesn't exist
    output_path = os.path.join(VIDEO_DIR, f"{video_id}.%(ext)s")
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=1080]',  # Best video-only, prefer mp4
        'outtmpl': output_path,
        'progress_hooks': [progress_hook],
        'quiet': False,  # Enable logs for debugging
        'noplaylist': True,
        'http_headers': {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            ),
        },
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(req.url, download=True)
            filename = ydl.prepare_filename(info).split('/')[-1]
        return {"status": "ok", "filename": filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}