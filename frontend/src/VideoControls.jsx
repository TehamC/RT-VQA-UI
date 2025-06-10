import React from 'react';

function VideoControls({ videoName, isPlaying, frameSkipRate, setShowSelectModal, handlePlayPause, handleFrameSkip, handleFrameSkipRateChange }) {
  return (
    <div className="controls">
      <button onClick={() => setShowSelectModal(true)}>
        {videoName ? `Selected: ${videoName}` : 'Select Video'}
      </button>
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
  );
}

export default VideoControls;