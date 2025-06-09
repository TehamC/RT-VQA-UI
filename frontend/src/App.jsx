import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('detections');
  const [detections, setDetections] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [videoName, setVideoName] = useState('');
  const [frameSkipRate, setFrameSkipRate] = useState(1);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [showSelectModal, setShowSelectModal] = useState(false);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [downloadProgress, setDownloadProgress] = useState({ progress: 0, speed: 0, eta: 0, status: 'idle' });

  const imageRef = useRef(null);
  const eventSourceRef = useRef(null);
  const chatEndRef = useRef(null);
  const wsRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  const extractYouTubeId = (url) => {
    const patterns = [
      /(?:v=|\/)([0-9A-Za-z_-]{11}).*/,
      /youtu\.be\/([0-9A-Za-z_-]{11})/
    ];
    for (const pattern of patterns) {
      const match = url.match(pattern);
      if (match) return match[1];
    }
    return null;
  };

  const startStreaming = (video, frameIndex) => {
    const url = `http://localhost:8000/stream/resume?video_name=${encodeURIComponent(video)}&frame_index=${frameIndex}&frame_skip=${frameSkipRate}`;
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const { image_base64, detections, frame_index } = data;

        if (imageRef.current && image_base64) {
          imageRef.current.src = `data:image/jpeg;base64,${image_base64}`;
        }

        setDetections(detections || []);
        setCurrentFrameIndex(frame_index);
      } catch (err) {
        console.error("Error parsing stream data:", err);
      }
    };

    eventSource.onerror = (err) => {
      console.error("SSE error:", err);
      eventSource.close();
      setIsPlaying(false);
    };

    setIsPlaying(true);
  };

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

    const videoId = extractYouTubeId(youtubeUrl);
    if (!videoId) {
      alert("Invalid YouTube URL.");
      return;
    }

    console.log(`Connecting WebSocket for video ID: ${videoId}`);
    wsRef.current = new WebSocket(`ws://localhost:8000/download_progress/${videoId}`);
    wsRef.current.onopen = () => console.log("WebSocket connected");
    wsRef.current.onmessage = (event) => {
      console.log("WebSocket message received:", event.data);
      const data = JSON.parse(event.data);
      setDownloadProgress(data);
      if (data.status === 'finished') {
        console.log("Download finished, closing WebSocket");
        wsRef.current.close();
      } else if (data.status === 'error') {
        console.log("Download error, closing WebSocket");
        wsRef.current.close();
        alert("Failed to download video.");
      }
    };
    wsRef.current.onerror = (err) => {
      console.error("WebSocket error:", err);
      wsRef.current.close();
      setDownloadProgress({ progress: 0, speed: 0, eta: 0, status: 'idle' });
      alert("Error connecting to download progress.");
    };
    wsRef.current.onclose = () => {
      console.log("WebSocket closed");
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

  const handlePlayPause = () => {
    if (isPlaying) {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      setIsPlaying(false);
    } else if (videoName) {
      startStreaming(videoName, currentFrameIndex);
    }
  };

  const handleFrameSkip = (offset) => {
    if (!videoName) return;
    const newIndex = Math.max(0, currentFrameIndex + offset);
    setCurrentFrameIndex(newIndex);

    if (isPlaying) {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      startStreaming(videoName, newIndex);
    } else {
      fetch(`http://localhost:8000/frame/infer?video_name=${encodeURIComponent(videoName)}&frame_index=${newIndex}`)
        .then((res) => res.json())
        .then((data) => {
          const { image_base64, detections, frame_index } = data;
          if (imageRef.current && image_base64) {
            imageRef.current.src = `data:image/jpeg;base64,${image_base64}`;
          }
          setDetections(detections || []);
          setCurrentFrameIndex(frame_index);
        })
        .catch((err) => console.error("Error fetching frame:", err));
    }
  };

  const handleFrameSkipRateChange = (e) => {
    setFrameSkipRate(parseInt(e.target.value, 10));
    if (videoName) {
      handleFrameSkip(0);
    }
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    const question = chatInput.trim();
    if (!question || !videoName) return;

    setChatMessages((prev) => [...prev, { text: question, sender: 'user' }]);
    setChatInput('');

    try {
      const res = await fetch("http://localhost:8000/set_llm_question", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      const data = await res.json();
      if (data.status === "ok") {
        setChatMessages((prev) => [...prev, { text: "LLM prompt updated.", sender: "llm" }]);
      } else {
        setChatMessages((prev) => [...prev, { text: `Error: ${data.message}`, sender: "llm" }]);
        return;
      }

      if (!isPlaying) {
        const inferRes = await fetch(`http://localhost:8000/frame/infer?video_name=${encodeURIComponent(videoName)}&frame_index=${currentFrameIndex}`);
        const inferData = await inferRes.json();

        if (inferData.image_base64) {
          imageRef.current.src = `data:image/jpeg;base64,${inferData.image_base64}`;
        }

        setDetections(inferData.detections || []);
        if (inferData.llm_answer) {
          setChatMessages((prev) => [...prev, { text: inferData.llm_answer, sender: "llm" }]);
        }
      }
    } catch (err) {
      console.error("Failed to submit prompt:", err);
      setChatMessages((prev) => [...prev, { text: "Error submitting prompt.", sender: "llm" }]);
    }
  };

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return (
    <div className="app">
      <div className="controls">
        <button onClick={() => setShowSelectModal(true)}>
          {videoName ? `Selected: ${videoName}` : 'Select Video'}
        </button>

        <input
          type="file"
          accept="video/mp4"
          onChange={handleUpload}
          id="file-input"
          style={{ display: 'none' }}
        />

        <button onClick={handlePlayPause}>{isPlaying ? 'Pause' : 'Play'}</button>
        <button onClick={() => handleFrameSkip(-300)}>-300</button>
        <button onClick={() => handleFrameSkip(300)}>+300</button>

        <label style={{ marginLeft: '10px' }}>FPS skip:</label>
        <select value={frameSkipRate} onChange={handleFrameSkipRateChange}>
          {[1, 5, 10, 20, 30, 60, 120, 240, 480].map((val) => (
            <option key={val} value={val}>{val}</option>
          ))}
        </select>
      </div>

      <div className="main-content">
        <div className="frames">
          {videoName ? (
            <img ref={imageRef} alt="Annotated frame" />
          ) : (
            <div className="placeholder">
              <span role="img" aria-label="video" style={{ fontSize: '4rem' }}>📹</span>
              <p>Select a video to start</p>
            </div>
          )}
        </div>
        <div className="chat-detections">
          <div className="tabs">
            <button onClick={() => setActiveTab('detections')}>Detections</button>
            <button onClick={() => setActiveTab('chat')}>Chat</button>
          </div>
          <div className="tab-content">
            {activeTab === 'detections' ? (
              <div>
                {detections.length > 0 ? (
                  detections.map((det, index) => (
                    <p key={index}>
                      {det.class_name}: {det.bbox.join(', ')} (Confidence: {det.confidence.toFixed(2)})
                    </p>
                  ))
                ) : (
                  'No Detections'
                )}
              </div>
            ) : (
              <div className="chat-area">
                <div className="chat-bubbles">
                  {chatMessages.map((msg, index) => (
                    <div key={index} className={`bubble ${msg.sender}`}>
                      {msg.text}
                    </div>
                  ))}
                  <div ref={chatEndRef} />
                </div>
                <form onSubmit={handleChatSubmit} className="chat-input">
                  <input
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Type your prompt..."
                  />
                </form>
              </div>
            )}
          </div>
        </div>
      </div>

      {showSelectModal && (
        <div className="modal-backdrop">
          <div className="modal">
            <h2>Select Video</h2>

            <div className="modal-section">
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
      )}
    </div>
  );
}

export default App;