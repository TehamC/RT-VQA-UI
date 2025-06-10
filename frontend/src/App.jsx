import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import VideoControls from './VideoControls';
import VideoPlayer from './VideoPlayer';
import ChatDetections from './ChatDetections';
import VideoModal from './VideoModal';

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

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) eventSourceRef.current.close();
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return (
    <div className="app">
      <VideoControls
        videoName={videoName}
        isPlaying={isPlaying}
        frameSkipRate={frameSkipRate}
        setShowSelectModal={setShowSelectModal}
        handlePlayPause={() => {
          if (isPlaying) {
            if (eventSourceRef.current) eventSourceRef.current.close();
            setIsPlaying(false);
          } else if (videoName) {
            const url = `http://localhost:8000/stream/resume?video_name=${encodeURIComponent(videoName)}&frame_index=${currentFrameIndex}&frame_skip=${frameSkipRate}`;
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

            eventSource.onerror = () => {
              eventSource.close();
              setIsPlaying(false);
            };

            setIsPlaying(true);
          }
        }}
        handleFrameSkip={(offset) => {
          if (!videoName) return;
          const newIndex = Math.max(0, currentFrameIndex + offset);
          setCurrentFrameIndex(newIndex);

          if (isPlaying) {
            if (eventSourceRef.current) eventSourceRef.current.close();
            const url = `http://localhost:8000/stream/resume?video_name=${encodeURIComponent(videoName)}&frame_index=${newIndex}&frame_skip=${frameSkipRate}`;
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

            eventSource.onerror = () => {
              eventSource.close();
              setIsPlaying(false);
            };

            setIsPlaying(true);
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
        }}
        handleFrameSkipRateChange={(e) => {
          setFrameSkipRate(parseInt(e.target.value, 10));
          if (videoName) {
            const newIndex = currentFrameIndex;
            if (isPlaying) {
              if (eventSourceRef.current) eventSourceRef.current.close();
              const url = `http://localhost:8000/stream/resume?video_name=${encodeURIComponent(videoName)}&frame_index=${newIndex}&frame_skip=${parseInt(e.target.value, 10)}`;
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

              eventSource.onerror = () => {
                eventSource.close();
                setIsPlaying(false);
              };

              setIsPlaying(true);
            }
          }
        }}
      />
      <div className="main-content">
        <VideoPlayer videoName={videoName} imageRef={imageRef} />
        <ChatDetections
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          detections={detections}
          chatMessages={chatMessages}
          chatInput={chatInput}
          setChatInput={setChatInput}
          chatEndRef={chatEndRef}
          handleChatSubmit={async (e) => {
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
          }}
        />
      </div>
      <VideoModal
        showSelectModal={showSelectModal}
        setShowSelectModal={setShowSelectModal}
        youtubeUrl={youtubeUrl}
        setYoutubeUrl={setYoutubeUrl}
        downloadProgress={downloadProgress}
        setDownloadProgress={setDownloadProgress}
        wsRef={wsRef}
        setVideoName={setVideoName}
        startStreaming={(video, frameIndex) => {
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

          eventSource.onerror = () => {
            eventSource.close();
            setIsPlaying(false);
          };

          setIsPlaying(true);
        }}
      />
    </div>
  );
}

export default App;