import React from 'react';

function VideoControls({ videoName, isPlaying, frameSkipRate, currentFrameIndex, totalFrames, setShowSelectModal, handlePlayPause, handleFrameSkip, handleFrameSkipRateChange, handleSliderChange }) {
  return (
    <div className="controls">
      <button onClick={handlePlayPause} style={{ padding: '5px 10px' }}>
        {isPlaying ? 'Pause' : 'Play'}
      </button>
      <button onClick={() => handleFrameSkip(-300)} style={{ padding: '5px 10px' }}>⏪ -60</button>
      <button onClick={() => handleFrameSkip(300)} style={{ padding: '5px 10px' }}>+60 ⏩</button>
      <input
        type="range"
        min="0"
        max="100"
        value={totalFrames ? (currentFrameIndex / totalFrames) * 100 : 0}
        onChange={handleSliderChange}
        style={{ width: '50%', margin: '0 10px' }}
      />
      <select value={frameSkipRate} onChange={handleFrameSkipRateChange} style={{ padding: '5px' }}>
        {[1, 5, 10, 20, 30, 60, 120, 240, 480].map((val) => (
          <option key={val} value={val}>{val} frames</option>
        ))}
      </select>
    </div>
  );
}

export default VideoControls;