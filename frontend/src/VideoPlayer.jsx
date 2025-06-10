import React from 'react';

function VideoPlayer({ videoName, imageRef }) {
  return (
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
  );
}

export default VideoPlayer;