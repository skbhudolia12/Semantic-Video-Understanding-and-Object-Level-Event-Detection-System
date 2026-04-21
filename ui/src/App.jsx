import { useState, useRef, useEffect } from 'react';
import { Camera, Upload, Play, Square, Video, ShieldAlert } from 'lucide-react';

function App() {
  const [streamId, setStreamId] = useState(null);
  const [useWebcam, setUseWebcam] = useState(false);
  const [videoFile, setVideoFile] = useState(null);
  const [query, setQuery] = useState('');
  const [compliance, setCompliance] = useState(true);
  const [attribute, setAttribute] = useState(false);
  
  const [logs, setLogs] = useState([]);
  const [violations, setViolations] = useState([]);
  
  const wsRef = useRef(null);
  const logsEndRef = useRef(null);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs, violations]);

  const handleStart = async () => {
    // 1. Upload Video if needed
    let videoParam = "0";
    if (!useWebcam && videoFile) {
      const formData = new FormData();
      formData.append('file', videoFile);
      const res = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      videoParam = data.path;
    }

    // 2. Generate a random stream ID
    const newStreamId = Math.random().toString(36).substring(7);
    
    // 3. Connect WebSocket
    const wsUrl = `ws://localhost:8000/ws/stats/${newStreamId}`;
    wsRef.current = new WebSocket(wsUrl);
    
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.status === "finished") {
        setStreamId(null);
        return;
      }
      
      // Construct a log entry
      if (data.detections && data.detections.length > 0) {
        setLogs(prev => [...prev, data]);
      }
      
      if (data.is_violation) {
        setViolations(prev => [...prev, data]);
      }
    };

    // 4. Construct Image SRC for MJPEG stream
    const params = new URLSearchParams({
      video_path: videoParam,
      query: query,
      compliance: compliance,
      attribute: attribute
    });
    setStreamId(`${newStreamId}?${params.toString()}`);
  };

  const handleStop = () => {
    setStreamId(null);
    if (wsRef.current) {
      wsRef.current.close();
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <Video className="text-primary-accent" size={32} color="#818cf8"/>
        <h1>Hybrid Pipeline Dashboard</h1>
      </header>

      <main className="dashboard">
        {/* Left Panel: Settings */}
        <aside className="glass-panel controls-panel">
          
          <div className="control-group">
            <label>Input Source</label>
            <label className="toggle-container">
              <input 
                type="checkbox" 
                checked={useWebcam} 
                onChange={(e) => setUseWebcam(e.target.checked)} 
              />
              <Camera size={18} /> Use Live Webcam
            </label>
            
            {!useWebcam && (
              <label className="toggle-container" style={{flexDirection: 'column', alignItems: 'flex-start'}}>
                <span style={{display: 'flex', gap: '0.5rem'}}><Upload size={18} /> Upload Video</span>
                <input 
                  type="file" 
                  accept="video/*" 
                  onChange={(e) => setVideoFile(e.target.files[0])} 
                  style={{width: '100%'}}
                />
              </label>
            )}
          </div>

          <div className="control-group">
            <label>Semantic Query (Optional)</label>
            <input 
              type="text" 
              placeholder="e.g. 'person in red jacket'" 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>

          <div className="control-group" style={{marginTop: '1rem'}}>
            <label>Pipeline Tiers</label>
            <label className="toggle-container">
              <input 
                type="checkbox" 
                checked={attribute} 
                onChange={(e) => setAttribute(e.target.checked)} 
              />
              Enable Attribute Matching
            </label>
            <label className="toggle-container">
              <input 
                type="checkbox" 
                checked={compliance} 
                onChange={(e) => setCompliance(e.target.checked)} 
              />
              <ShieldAlert size={18} /> Enable Neural Compliance
            </label>
          </div>

          <div style={{marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: '0.5rem'}}>
            {!streamId ? (
              <button className="btn" onClick={handleStart}>
                <Play size={18} /> Start Pipeline
              </button>
            ) : (
              <button className="btn stop" onClick={handleStop}>
                <Square size={18} /> Stop Stream
              </button>
            )}
          </div>
        </aside>

        {/* Center Panel: Stream */}
        <section className="glass-panel video-panel">
          {streamId ? (
            <>
              <div className="live-indicator">LIVE</div>
              <img 
                src={`http://localhost:8000/video_feed/${streamId}`} 
                alt="Video Stream" 
                className="video-frame"
              />
            </>
          ) : (
            <div className="video-placeholder">
              <Video size={64} style={{ opacity: 0.2 }} />
              <p>Configure settings and start the pipeline.</p>
            </div>
          )}
        </section>

        {/* Right Panel: Watch/Log */}
        <aside className="glass-panel logs-panel">
          
          <div className="log-section" style={{flex: 0.3}}>
            <h3>Violations <span className="badge">{violations.length}</span></h3>
            <div className="log-list">
              {violations.map((v, i) => (
                <div key={i} className="detection-card violation">
                  <div className="det-header">
                    <strong>Rule Check Failed</strong>
                  </div>
                  <div className="det-meta">
                    <span>{v.timestamp}s</span>
                    <span>Conf: {v.violation_confidence}</span>
                  </div>
                </div>
              ))}
              {violations.length === 0 && <p style={{opacity: 0.5, textAlign: 'center', marginTop: '1rem'}}>All clear</p>}
              <div ref={logsEndRef} />
            </div>
          </div>

          <div className="log-section" style={{flex: 0.7}}>
            <h3>Live Objects Tracker <span className="badge">{logs.length}</span></h3>
            <div className="log-list">
              {logs.map((log, i) => (
                <div key={i} style={{marginBottom: '0.5rem'}}>
                  {log.detections.map((det, j) => (
                    <div key={j} className="detection-card">
                      <div className="det-header">
                        <strong>{det.label}</strong>
                        {det.track_id !== -1 && <span className="track-id">T#{det.track_id}</span>}
                      </div>
                      <div className="det-meta">
                        <span>{log.timestamp}s</span>
                        {det.similarity && <span>Sim: {det.similarity}</span>}
                      </div>
                    </div>
                  ))}
                </div>
              ))}
              {logs.length === 0 && <p style={{opacity: 0.5, textAlign: 'center', marginTop: '3rem'}}>Waiting for detections...</p>}
              <div ref={logsEndRef} />
            </div>
          </div>

        </aside>
      </main>
    </div>
  );
}

export default App;
