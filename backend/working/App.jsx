import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('detections');
  const [detections, setDetections] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [videoName, setVideoName] = useState('');
  const imageRef = useRef(null);
  const eventSourceRef = useRef(null);

  const handleUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'video/mp4') {
      setVideoName(file.name);
      fetch(`http://localhost:8000/stream?video_name=${file.name}`, { method: 'GET' })
        .then(() => setIsPlaying(true));
    }
  };

  const handlePlayStop = () => {
    if (isPlaying) {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      setIsPlaying(false);
    } else {
      setIsPlaying(true);
    }
  };

  const handleChatSubmit = (e) => {
    e.preventDefault();
    if (chatInput.trim()) {
      setChatMessages([...chatMessages, { text: chatInput, sender: 'user' }]);
      setChatMessages([...chatMessages, { text: 'Dummy reply text', sender: 'llm' }]);
      setChatInput('');
    }
  };

  useEffect(() => {
    if (isPlaying && videoName) {
      const url = `http://localhost:8000/stream?video_name=${videoName}`;
      const eventSource = new EventSource(url);
      eventSourceRef.current = eventSource;

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const { image_base64, detections } = data;

          if (imageRef.current && image_base64) {
            imageRef.current.src = `data:image/jpeg;base64,${image_base64}`;
          }

          setDetections(detections || []);
        } catch (err) {
          console.error("Error parsing stream data:", err);
        }
      };

      eventSource.onerror = (err) => {
        console.error("SSE error:", err);
        eventSource.close();
        setIsPlaying(false);
      };
    }

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [isPlaying, videoName]);

  return (
    <div className="app">
      <div className="controls">
        <input type="file" accept="video/mp4" onChange={handleUpload} id="select" style={{ display: 'none' }} />
        <label htmlFor="select">{videoName ? `Select: ${videoName}` : 'Select'}</label>
        <button onClick={handlePlayStop}>{isPlaying ? 'Stop' : 'Play'}</button>
      </div>
      <div className="main-content">
        <div className="frames">
          <img ref={imageRef} alt="Annotated frame" />
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
                      {det.class_name}: {det.bbox.join(', ')} (Confidence: {det.confidence})
                    </p>
                  ))
                ) : (
                  'No Detections'
                )}
              </div>
            ) : (
              <div>
                <form onSubmit={handleChatSubmit}>
                  <input
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Type your prompt..."
                  />
                </form>
                <div className="chat-bubbles">
                  {chatMessages.map((msg, index) => (
                    <div key={index} className={`bubble ${msg.sender}`}>
                      {msg.text}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;