import os
import re
from typing import Dict
import yt_dlp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from config import VIDEO_DIR
import threading
import logging
import psutil
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class YouTubeRequest(BaseModel):
    url: str

download_progress: Dict[str, Dict] = {}
ydl_instances: Dict[str, yt_dlp.YoutubeDL] = {}
download_threads: Dict[str, threading.Thread] = {}
download_locks: Dict[str, threading.Lock] = {}
subprocess_pids: Dict[str, set] = {}

def extract_youtube_id(url: str) -> str:
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
    try:
        video_id = d.get('info_dict', {}).get('id')
        if not video_id:
            logger.warning("No video ID in progress hook")
            return
        if download_progress.get(video_id, {}).get('cancel_download', False):
            logger.info(f"Cancellation requested for {video_id} in progress_hook")
            raise yt_dlp.utils.DownloadError("Download cancelled by user")
        if d['status'] == 'downloading':
            downloaded_bytes = d.get('downloaded_bytes', 0)
            total_bytes = d.get('total_bytes', d.get('total_bytes_estimate', downloaded_bytes))
            speed = d.get('speed', 0) or 0
            eta = d.get('eta', 0) or 0
            progress = (downloaded_bytes / total_bytes) * 100 if total_bytes > 0 else 0
            download_progress[video_id] = {
                'progress': min(progress, 100),
                'speed': speed / 1024 / 1024,
                'eta': eta,
                'status': 'downloading',
                'total_bytes': total_bytes,
                'cancel_download': False
            }
            # logger.info(f"Progress update for {video_id}: {progress:.2f}%")
        elif d['status'] == 'finished':
            download_progress[video_id] = {
                'progress': 100,
                'speed': 0,
                'eta': 0,
                'status': 'downloading',  # Delay 'finished' until post-processing
                'total_bytes': d.get('total_bytes', d.get('total_bytes_estimate', 0)),
                'cancel_download': False
            }
            logger.info(f"yt-dlp download finished for {video_id}")
        elif d['status'] == 'error':
            download_progress[video_id] = {
                'progress': 0,
                'speed': 0,
                'eta': 0,
                'status': 'error',
                'total_bytes': 0,
                'cancel_download': False,
                'error': 'yt-dlp download failed'
            }
            logger.error(f"yt-dlp download error for {video_id}")
    except yt_dlp.utils.DownloadError:
        logger.info(f"Download cancelled for {video_id}")
        download_progress[video_id] = {
            'progress': 0,
            'speed': 0,
            'eta': 0,
            'status': 'canceled',
            'total_bytes': 0,
            'cancel_download': True
        }
        raise
    except Exception as e:
        logger.error(f"Error in progress_hook: {str(e)}")
        download_progress[video_id] = {
            'progress': 0,
            'speed': 0,
            'eta': 0,
            'status': 'error',
            'total_bytes': 0,
            'cancel_download': False,
            'error': str(e)
        }

def cleanup_part_files(video_id: str, preserve_final: bool = True, preserve_temp: bool = True) -> None:
    part_files = [f for f in os.listdir(VIDEO_DIR) if video_id in f and ('part' in f.lower() or '_temp' in f)]
    final_file = f"{video_id}.mp4"
    for part_file in part_files:
        if (preserve_final and part_file == final_file) or (preserve_temp and '_temp' in part_file):
            continue
        try:
            file_path = os.path.join(VIDEO_DIR, part_file)
            os.remove(file_path)
            logger.info(f"Removed partial file: {part_file}")
        except Exception as e:
            logger.error(f"Error removing partial file {part_file}: {str(e)}")

def kill_subprocesses(video_id: str) -> None:
    if video_id in subprocess_pids:
        for pid in subprocess_pids[video_id]:
            try:
                process = psutil.Process(pid)
                process.terminate()
                process.wait(timeout=3)
                logger.info(f"Terminated subprocess {pid} for {video_id}")
            except psutil.NoSuchProcess:
                logger.info(f"Subprocess {pid} for {video_id} already terminated")
            except Exception as e:
                logger.error(f"Error terminating subprocess {pid}: {str(e)}")
        subprocess_pids.pop(video_id, None)

def postprocess_video(input_path: str, output_path: str) -> None:
    if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
        logger.error(f"Input file {input_path} is missing or empty")
        raise Exception("Input file invalid")
    try:
        logger.info(f"Starting post-processing: {input_path} -> {output_path}")
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-hwaccel', 'cuda', '-i', input_path,
            '-vf', 'scale=w=iw*min(896/iw\,896/ih):h=ih*min(896/iw\,896/ih),pad=896:896:(896-iw*min(896/iw\,896/ih))/2:(896-ih*min(896/iw\,896/ih))/2:black',
            '-c:v', 'h264_nvenc', '-preset', 'fast', '-b:v', '4M',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            output_path
        ]
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        logger.info(f"Postprocessed video: {output_path}, ffmpeg stdout: {result.stdout}")
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logger.error(f"Output file {output_path} is empty or missing")
            raise Exception("Post-processing failed: empty output")
        os.remove(input_path)
        logger.info(f"Removed original file: {input_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        raise Exception(f"FFmpeg failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Postprocessing error: {str(e)}")
        raise

def download_video(url: str, ydl_opts: Dict, video_id: str) -> None:
    logger.info(f"Starting download thread for {video_id}")
    temp_path = os.path.join(VIDEO_DIR, f"{video_id}_temp.mp4")
    final_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    try:
        lock = download_locks.setdefault(video_id, threading.Lock())
        if not lock.acquire(blocking=False):
            logger.warning(f"Download already in progress for {video_id}")
            download_progress[video_id] = {
                'progress': 0,
                'speed': 0,
                'eta': 0,
                'status': 'error',
                'total_bytes': 0,
                'cancel_download': False,
                'error': 'Download already in progress'
            }
            return
        try:
            ydl_opts['outtmpl'] = temp_path
            ydl_opts['retries'] = 3
            ydl_opts['fragment_retries'] = 3
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl_instances[video_id] = ydl
                download_progress[video_id] = {'progress': 0, 'speed': 0, 'eta': 0, 'status': 'downloading', 'total_bytes': 0, 'cancel_download': False}
                subprocess_pids[video_id] = {p.pid for p in psutil.process_iter() if 'yt-dlp' in p.name().lower()}
                try:
                    ydl.download([url])
                    logger.info(f"yt-dlp download completed for {video_id}")
                except Exception as e:
                    logger.error(f"yt-dlp download failed for {video_id}: {str(e)}")
                    raise
                if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                    logger.error(f"Temporary file {temp_path} is missing or empty")
                    download_progress[video_id] = {
                        'progress': 0,
                        'speed': 0,
                        'eta': 0,
                        'status': 'error',
                        'total_bytes': 0,
                        'cancel_download': False,
                        'error': 'Temporary file empty or missing'
                    }
                    raise Exception("Download failed: empty temporary file")
                postprocess_video(temp_path, final_path)
                download_progress[video_id] = {
                    'progress': 100,
                    'speed': 0,
                    'eta': 0,
                    'status': 'finished',
                    'total_bytes': os.path.getsize(final_path),
                    'cancel_download': False
                }
                logger.info(f"Download and post-processing completed for {video_id}")
        except Exception as e:
            logger.error(f"Download or postprocessing failed for {video_id}: {str(e)}")
            download_progress[video_id] = {
                'progress': 0,
                'speed': 0,
                'eta': 0,
                'status': 'error',
                'total_bytes': 0,
                'cancel_download': False,
                'error': str(e)
            }
            cleanup_part_files(video_id, preserve_final=True, preserve_temp=True)
            raise
        finally:
            lock.release()
            ydl_instances.pop(video_id, None)
            download_threads.pop(video_id, None)
            kill_subprocesses(video_id)
    except Exception as e:
        logger.error(f"Error in download_video for {video_id}: {str(e)}")
        download_progress[video_id] = {
            'progress': 0,
            'speed': 0,
            'eta': 0,
            'status': 'error',
            'total_bytes': 0,
            'cancel_download': False,
            'error': str(e)
        }
        cleanup_part_files(video_id, preserve_final=True, preserve_temp=True)

@router.post("/download_youtube")
async def download_youtube(req: YouTubeRequest):
    video_id = extract_youtube_id(req.url)
    if not video_id:
        logger.error(f"Invalid YouTube URL: {req.url}")
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    final_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
        logger.info(f"Video {video_id} already exists")
        download_progress[video_id] = {
            'progress': 100,
            'speed': 0,
            'eta': 0,
            'status': 'finished',
            'total_bytes': os.path.getsize(final_path),
            'cancel_download': False
        }
        return {"status": "ok", "video_id": video_id}
    
    ydl_opts = {
        'format': "bestvideo[height<=1080][ext=mp4]/bestvideo[height<=1080]/best[height<=1080]",
        'progress_hooks': [progress_hook],
        'quiet': False,
        'no_warnings': False,
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4'
        }],
        'logger': logger
    }
    
    download_progress[video_id] = {'progress': 0, 'speed': 0, 'eta': 0, 'status': 'downloading', 'total_bytes': 0, 'cancel_download': False}
    thread = threading.Thread(target=download_video, args=(req.url, ydl_opts, video_id))
    download_threads[video_id] = thread
    thread.start()
    
    logger.info(f"Started download for {video_id}")
    return {"status": "ok", "video_id": video_id}

@router.post("/cancel_download/{video_id}")
async def cancel_download(video_id: str):
    logger.info(f"Received cancel request for {video_id}")
    if video_id not in download_progress:
        logger.warning(f"No download found for {video_id}")
        raise HTTPException(status_code=404, detail="No download found")
    
    download_progress[video_id]['cancel_download'] = True
    kill_subprocesses(video_id)
    cleanup_part_files(video_id, preserve_final=True, preserve_temp=True)
    
    if video_id in ydl_instances:
        try:
            ydl_instances[video_id].close()
        except Exception as e:
            logger.error(f"Error closing yt-dlp instance for {video_id}: {str(e)}")
    
    download_progress[video_id] = {
        'progress': 0,
        'speed': 0,
        'eta': 0,
        'status': 'canceled',
        'total_bytes': 0,
        'cancel_download': False
    }
    logger.info(f"Download canceled for {video_id}")
    return {"status": "ok", "message": f"Download canceled for {video_id}"}

@router.get("/verify_file/{video_id}")
async def verify_file(video_id: str):
    file_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    exists = os.path.exists(file_path)
    size = os.path.getsize(file_path) if exists else 0
    logger.info(f"Verifying file {file_path}: exists={exists}, size={size}")
    return {"exists": exists, "size": size}