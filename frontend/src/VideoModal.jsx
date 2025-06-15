import React, { useState, useEffect } from 'react';

function VideoModal({ showSelectModal, setShowSelectModal, youtubeUrl, setYoutubeUrl, downloadProgress, setDownloadProgress, wsRef, setVideoName, startStreaming }) {
  const [videoId, setVideoId] = useState(null);
  const [isCancelRequested, setIsCancelRequested] = useState(false);

  const handleUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'video/mp4') {
      setVideoName(file.name.replace(/\.mp4$/, '')); // Strip .mp4
      setShowSelectModal(false);
      startStreaming(file.name.replace(/\.mp4$/, ''), 0);
    } else {
      alert('Please select a valid .mp4 file.');
    }
  };

  const handleCancelDownload = async () => {
    if (!videoId) {
      console.log("No videoId for cancellation");
      return;
    }
    console.log(`User requested cancel for videoId: ${videoId}`);
    setIsCancelRequested(true);
    try {
      const res = await fetch(`http://localhost:8000/cancel_download/${videoId}`, { method: 'POST' });
      const data = await res.json();
      console.log("Cancel response:", data);
      if (!res.ok) throw new Error(data.message || `Status: ${res.status}`);
      setDownloadProgress({ progress: 0, speed: 0, eta: 0, status: 'canceled' });
      setShowSelectModal(false);
      setYoutubeUrl('');
      setVideoId(null);
    } catch (err) {
      console.error("Cancel download error:", err.message);
      setDownloadProgress({ ...downloadProgress, status: 'error', error: err.message });
    } finally {
      wsRef.current?.close();
      setIsCancelRequested(false);
    }
  };

  const handleYouTubeLoad = async () => {
    if (!youtubeUrl.trim()) {
      console.log("Empty YouTube URL");
      alert("Please enter a YouTube URL.");
      return;
    }

    const pattern = /([0-9A-Za-z_-]{11})/;
    const match = youtubeUrl.match(pattern);
    const newVideoId = match ? match[1] : null;

    if (!newVideoId) {
      console.log("Invalid YouTube URL");
      alert("Invalid YouTube URL.");
      return;
    }

    console.log(`Starting download for videoId: ${newVideoId}`);
    setVideoId(newVideoId);
    setDownloadProgress({ progress: 0, speed: 0, eta: 0, status: 'pending' });

    try {
      console.log("Fetching /download_youtube...");
      const res = await fetch("http://localhost:8000/download_youtube", {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: youtubeUrl }),
      });

      console.log("Load YouTube response status:", res.status);
      const data = await res.json();
      console.log("Load YouTube response:", data);

      if (!res.ok) throw new Error(data.detail || `Status: ${res.status}`);
      if (data.status !== 'ok') throw new Error('Download initiation failed');
    } catch (err) {
      console.error("YouTube load error:", err.message);
      alert("Error loading YouTube video: " + err.message);
      setDownloadProgress({ progress: 0, speed: 0, eta: 0, status: 'error', error: err.message });
      setVideoId(null);
      setShowSelectModal(false);
    }
  };

  useEffect(() => {
    if (!videoId) return;

    console.log(`Connecting WebSocket for videoId: ${videoId}`);
    wsRef.current = new WebSocket(`ws://localhost:8000/download_progress/${videoId}`);

    wsRef.current.onopen = () => {
      console.log(`WebSocket opened for videoId: ${videoId}`);
    };

    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("WebSocket received progress:", data);
        console.log("Current downloadProgress state:", downloadProgress);
        if (!isCancelRequested) {
          setDownloadProgress(data);
          console.log("Updated downloadProgress:", data);
          if (data.status === 'finished' && data.progress >= 100) {
            fetch(`http://localhost:8000/verify_file/${videoId}`)
              .then(res => res.json())
              .then(fileData => {
                if (fileData.exists && fileData.size > 0) {
                  setVideoName(videoId); // Already without .mp4
                  startStreaming(videoId, 0);
                  setShowSelectModal(false);
                  setYoutubeUrl('');
                  setVideoId(null);
                } else {
                  throw new Error('Downloaded file is empty or missing');
                }
              })
              .catch(err => {
                console.error("File verification error:", err.message);
                setDownloadProgress({ ...data, status: 'error', error: 'Downloaded file is empty or missing' });
                setShowSelectModal(false);
                setVideoId(null);
              });
          } else if (data.status === 'error') {
            setDownloadProgress({ ...data, status: 'error', error: data.error || 'Download failed' });
            setShowSelectModal(false);
            setVideoId(null);
          }
        }
      } catch (err) {
        console.error("WebSocket message error:", err);
      }
    };

    wsRef.current.onerror = (err) => {
      console.error("WebSocket error:", err);
      if (!isCancelRequested && downloadProgress.status !== 'finished') {
        setDownloadProgress({ progress: 0, speed: 0, eta: 0, status: 'error', error: 'Download progress connection failed' });
        setShowSelectModal(false);
        setVideoId(null);
      }
    };

    wsRef.current.onclose = () => {
      console.log(`WebSocket closed for videoId: ${videoId}`);
    };

    return () => {
      console.log("Cleaning up WebSocket");
      wsRef.current?.close();
    };
  }, [videoId, isCancelRequested, downloadProgress.status, wsRef]);

  if (!showSelectModal) return null;

  return (
    <div className="modal-backdrop">
      <div className="modal">
        <h2>Select Video</h2>
        <div className="modal-section">
          <input
            type="file"
            accept="video/mp4"
            onChange={handleUpload}
            id="file-input"
            style={{ display: 'none' }}
          />
          <label htmlFor="file-input" className="button">📁 Upload .mp4 File</label>
        </div>
        <div className="modal-section">
          <input
            type="text"
            placeholder="Paste YouTube URL"
            value={youtubeUrl}
            onChange={(e) => setYoutubeUrl(e.target.value)}
          />
          <button onClick={handleYouTubeLoad} disabled={downloadProgress.status === 'downloading' || downloadProgress.status === 'pending'}>
            📹 Load YouTube Video
          </button>
        </div>
        {(downloadProgress.status === 'downloading' || downloadProgress.status === 'pending') && (
          <div className="progress-section">
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${downloadProgress.progress}%` }}></div>
            </div>
            <p>
              Downloading: {downloadProgress.progress.toFixed(1)}% | 
              Speed: {downloadProgress.speed.toFixed(2)} MB/s | 
              ETA: {Math.floor(Math.round(downloadProgress.eta) / 60)}m {Math.round(downloadProgress.eta) % 60}s
            </p>
            <button onClick={handleCancelDownload} disabled={isCancelRequested}>
              Cancel Download
            </button>
          </div>
        )}
        {downloadProgress.status === 'error' && (
          <p className="error">Error: {downloadProgress.error || 'Download failed'}</p>
        )}
        <div className="modal-section">
          <button onClick={() => setShowSelectModal(false)}>Close</button>
        </div>
      </div>
    </div>
  );
}

export default VideoModal;