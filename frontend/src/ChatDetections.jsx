import React from 'react';

function ChatDetections({ activeTab, setActiveTab, detections, chatMessages, chatInput, setChatInput, chatEndRef, handleChatSubmit }) {
  return (
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
  );
}

export default ChatDetections;