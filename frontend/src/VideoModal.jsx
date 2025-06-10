import React from 'react';

function VideoModal({ showSelectModal, setShowSelectModal, youtubeUrl, setYoutubeUrl, downloadProgress, setDownloadProgress, wsRef, setVideoName, startStreaming }) {
  const handleUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'video/mp4') {
      setVideoName(file.name);
      setShowSelectModal(false);
      startStreaming(file.name, 0);
    }
  };

  const handleYouTubeLoad = async () => {
    if (!youtubeUrl.trim()) {
      alert("Please enter a YouTube URL.");
      return;
    }

    const videoId = youtubeUrl.match(/(?:v=|\/)([0-9A-Za-z_-]{11}).*|$/)?.[1] || youtubeUrl.match(/youtu\.be\/([0-9A-Za-z_-]{11})/)?.[1];
    if (!videoId) {
      alert("Invalid YouTube URL.");
      return;
    }

    wsRef.current = new WebSocket(`ws://localhost:8000/download_progress/${videoId}`);
    wsRef.current.onopen = () => console.log("WebSocket connected");
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setDownloadProgress(data);
      if (data.status === 'finished') {
        wsRef.current.close();
      } else if (data.status === 'error') {
        wsRef.current.close();
        alert("Failed to download video.");
      }
    };
    wsRef.current.onerror = () => {
      wsRef.current.close();
      setDownloadProgress({ progress: 0, speed: 0, eta: 0, status: 'idle' });
      alert("Error connecting to download progress.");
    };
    wsRef.current.onclose = () => {
      setDownloadProgress({ progress: 0, speed: 0, eta: 0, status: 'idle' });
    };

    try {
      const res = await fetch("http://localhost:8000/load_youtube", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: youtubeUrl }),
      });

      const data = await res.json();
      if (data.status === "ok") {
        setVideoName(data.filename);
        startStreaming(data.filename, 0);
        setShowSelectModal(false);
        setYoutubeUrl('');
      } else {
        alert(`Failed to load video: ${data.message}`);
        wsRef.current.close();
      }
    } catch (err) {
      console.error("YouTube load error:", err);
      alert("Error loading YouTube video.");
      wsRef.current.close();
    }
  };

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
          <button onClick={handleYouTubeLoad} disabled={downloadProgress.status === 'downloading'}>
            📹 Load YouTube Video
          </button>
        </div>
        {downloadProgress.status === 'downloading' && (
          <div className="progress-section">
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${downloadProgress.progress}%` }}
              ></div>
            </div>
            <p>
              Downloading: {downloadProgress.progress.toFixed(1)}% | 
              Speed: {downloadProgress.speed.toFixed(2)} MB/s | 
              ETA: {Math.floor(downloadProgress.eta / 60)}m {downloadProgress.eta % 60}s
            </p>
          </div>
        )}
        <div className="modal-section">
          <button onClick={() => setShowSelectModal(false)}>Cancel</button>
        </div>
      </div>
    </div>
  );
}

export default VideoModal;