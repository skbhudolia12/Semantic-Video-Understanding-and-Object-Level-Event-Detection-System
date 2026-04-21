import { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, Upload, Play, Square, Video, Plus, X, Loader2, ShieldAlert, Eye } from 'lucide-react';

const API = 'http://localhost:8000';

// How many event log entries to keep
const MAX_LOG = 80;

function RuleItem({ rule, onRemove, matchState }) {
  const dot = matchState?.matched
    ? (rule.is_violation ? '#ef4444' : '#10b981')
    : 'rgba(255,255,255,0.15)';

  return (
    <div className="rule-item" style={{ borderLeftColor: rule.color_hex }}>
      <div className="rule-item-header">
        <span className="rule-dot" style={{ background: dot }} />
        <span className="rule-label">{rule.display_label}</span>
        <span className="rule-id-badge">{rule.rule_id}</span>
        <button className="rule-remove" onClick={() => onRemove(rule.rule_id)}>
          <X size={13} />
        </button>
      </div>
      <div className="rule-original">{rule.original_text}</div>
    </div>
  );
}

function ParseItem({ text }) {
  return (
    <div className="rule-item parsing">
      <div className="rule-item-header">
        <Loader2 size={13} className="spin" />
        <span className="rule-original" style={{ color: 'var(--text-secondary)' }}>{text}</span>
      </div>
    </div>
  );
}

export default function App() {
  // Stream state
  const [streamId, setStreamId]   = useState(null);
  const [useWebcam, setUseWebcam] = useState(false);
  const [videoFile, setVideoFile] = useState(null);

  // Rule list: [{rule_id, display_label, original_text, color_hex, is_violation}, ...]
  const [rules, setRules]           = useState([]);
  // Per-rule current match state keyed by rule_id
  const [matchStates, setMatchStates] = useState({});
  // Items currently being parsed (show spinner)
  const [parsing, setParsing]       = useState([]);
  // Event log [{timestamp, rule_id, display_label, is_violation, color_hex}]
  const [eventLog, setEventLog]     = useState([]);
  // Rule text input
  const [ruleInput, setRuleInput]   = useState('');

  const wsRef      = useRef(null);
  const logEndRef  = useRef(null);

  // Fetch existing rules on mount
  useEffect(() => {
    fetch(`${API}/rules`)
      .then(r => r.json())
      .then(setRules)
      .catch(() => {});
  }, []);

  // Auto-scroll event log
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [eventLog]);

  // ------------------------------------------------------------------
  // Rule management
  // ------------------------------------------------------------------

  const handleAddRule = useCallback(async () => {
    const text = ruleInput.trim();
    if (!text) return;
    setRuleInput('');
    setParsing(prev => [...prev, text]);

    try {
      const res = await fetch(`${API}/rules`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(`Rule parse failed: ${err.detail}`);
        return;
      }
      const rule = await res.json();
      setRules(prev => [...prev, rule]);
    } catch (e) {
      alert(`Network error: ${e.message}`);
    } finally {
      setParsing(prev => prev.filter(t => t !== text));
    }
  }, [ruleInput]);

  const handleRemoveRule = useCallback(async (rule_id) => {
    await fetch(`${API}/rules/${rule_id}`, { method: 'DELETE' });
    setRules(prev => prev.filter(r => r.rule_id !== rule_id));
    setMatchStates(prev => { const s = { ...prev }; delete s[rule_id]; return s; });
    setEventLog(prev => prev.filter(ev => ev.rule_id !== rule_id));
  }, []);

  // ------------------------------------------------------------------
  // Stream control
  // ------------------------------------------------------------------

  const handleStart = async () => {
    let videoParam = '0';
    if (!useWebcam && videoFile) {
      const form = new FormData();
      form.append('file', videoFile);
      const res  = await fetch(`${API}/upload`, { method: 'POST', body: form });
      const data = await res.json();
      videoParam = data.path;
    }

    const sid = Math.random().toString(36).substring(7);

    // WebSocket for per-frame metadata
    const ws = new WebSocket(`ws://localhost:8000/ws/stats/${sid}`);
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data);
      if (data.status === 'finished') { setStreamId(null); return; }

      const { rule_results = [], timestamp } = data;

      // Update per-rule match state
      const newStates = {};
      for (const r of rule_results) {
        newStates[r.rule_id] = r;
      }
      setMatchStates(prev => ({ ...prev, ...newStates }));

      // Append to event log for every matched rule
      const events = rule_results.filter(r => r.matched).map(r => ({
        timestamp,
        rule_id: r.rule_id,
        display_label: r.display_label,
        is_violation: r.is_violation,
        color_hex: r.color_hex,
      }));
      if (events.length > 0) {
        setEventLog(prev => [...prev.slice(-MAX_LOG), ...events]);
      }
    };

    const params = new URLSearchParams({ video_path: videoParam, fps: 30 });
    setStreamId(`${sid}?${params}`);
    setEventLog([]);
    setMatchStates({});
  };

  const handleStop = () => {
    setStreamId(null);
    wsRef.current?.close();
  };

  const activeViolations = Object.values(matchStates).filter(m => m.matched && m.is_violation).length;

  return (
    <div className="app-container">
      <header className="header">
        <Video size={32} color="#818cf8" />
        <h1>Hybrid Pipeline Dashboard</h1>
        {activeViolations > 0 && (
          <span className="global-violation-badge">
            <ShieldAlert size={14} /> {activeViolations} active violation{activeViolations > 1 ? 's' : ''}
          </span>
        )}
      </header>

      <main className="dashboard">
        {/* ---- Left Panel: Source + Rules ---- */}
        <aside className="glass-panel controls-panel">

          {/* Input source */}
          <div className="control-group">
            <label>Input Source</label>
            <label className="toggle-container">
              <input type="checkbox" checked={useWebcam}
                onChange={e => setUseWebcam(e.target.checked)} />
              <Camera size={18} /> Use Live Webcam
            </label>
            {!useWebcam && (
              <label className="toggle-container" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                <span style={{ display: 'flex', gap: '0.5rem' }}><Upload size={18} /> Upload Video</span>
                <input type="file" accept="video/*"
                  onChange={e => setVideoFile(e.target.files[0])}
                  style={{ width: '100%' }} />
              </label>
            )}
          </div>

          {/* Rule input */}
          <div className="control-group" style={{ flex: 1 }}>
            <label>Detection Rules</label>
            <p style={{ fontSize: '0.78rem', lineHeight: 1.5 }}>
              Describe what to detect or flag in plain English. Rules are parsed by AI and evaluated live.
            </p>

            <div className="rule-input-row">
              <input
                type="text"
                placeholder="e.g. person on a bike without a helmet"
                value={ruleInput}
                onChange={e => setRuleInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleAddRule()}
                style={{ flex: 1 }}
              />
              <button className="btn icon-btn" onClick={handleAddRule}
                disabled={!ruleInput.trim() || parsing.length > 0}>
                <Plus size={16} />
              </button>
            </div>

            {/* Active rule list */}
            <div className="rule-list">
              {rules.map(rule => (
                <RuleItem
                  key={rule.rule_id}
                  rule={rule}
                  onRemove={handleRemoveRule}
                  matchState={matchStates[rule.rule_id]}
                />
              ))}
              {parsing.map((text, i) => <ParseItem key={i} text={text} />)}
              {rules.length === 0 && parsing.length === 0 && (
                <p style={{ opacity: 0.4, fontSize: '0.78rem', textAlign: 'center', marginTop: '1rem' }}>
                  No rules yet — add one above.
                </p>
              )}
            </div>
          </div>

          {/* Stream control */}
          <div>
            {!streamId ? (
              <button className="btn" style={{ width: '100%' }} onClick={handleStart}>
                <Play size={18} /> Start Stream
              </button>
            ) : (
              <button className="btn stop" style={{ width: '100%' }} onClick={handleStop}>
                <Square size={18} /> Stop Stream
              </button>
            )}
          </div>
        </aside>

        {/* ---- Center Panel: Video ---- */}
        <section className="glass-panel video-panel">
          {streamId ? (
            <>
              <div className="live-indicator">LIVE</div>
              <img
                src={`${API}/video_feed/${streamId}`}
                alt="Video Stream"
                className="video-frame"
              />
            </>
          ) : (
            <div className="video-placeholder">
              <Video size={64} style={{ opacity: 0.2 }} />
              <p>Add rules and start the stream.</p>
            </div>
          )}
        </section>

        {/* ---- Right Panel: Rule status + Event log ---- */}
        <aside className="glass-panel logs-panel">

          {/* Rule status cards */}
          <div className="log-section" style={{ flex: '0 0 auto', maxHeight: '45%' }}>
            <h3>
              <Eye size={14} style={{ marginRight: '0.4rem' }} />
              Active Rules
              <span className="badge" style={{ marginLeft: 'auto' }}>{rules.length}</span>
            </h3>
            <div className="log-list">
              {rules.map(rule => {
                const state = matchStates[rule.rule_id];
                const matched = state?.matched;
                const violation = matched && rule.is_violation;
                return (
                  <div key={rule.rule_id}
                    className={`detection-card${violation ? ' violation' : ''}`}
                    style={{ borderLeftColor: rule.color_hex }}>
                    <div className="det-header">
                      <strong>{rule.display_label}</strong>
                      <span className="rule-id-badge">{rule.rule_id}</span>
                    </div>
                    <div className="det-meta">
                      <span>{matched ? (violation ? 'VIOLATION' : 'MATCHED') : 'watching...'}</span>
                    </div>
                  </div>
                );
              })}
              {rules.length === 0 && (
                <p style={{ opacity: 0.4, textAlign: 'center', marginTop: '1rem', fontSize: '0.8rem' }}>
                  No rules loaded
                </p>
              )}
            </div>
          </div>

          {/* Event log */}
          <div className="log-section" style={{ flex: 1, minHeight: 0 }}>
            <h3>
              Event Log
              <span className="badge" style={{ marginLeft: 'auto' }}>{eventLog.length}</span>
            </h3>
            <div className="log-list">
              {eventLog.slice().reverse().map((ev, i) => (
                <div key={i}
                  className={`detection-card${ev.is_violation ? ' violation' : ''}`}
                  style={{ borderLeftColor: ev.color_hex }}>
                  <div className="det-header">
                    <strong>{ev.display_label}</strong>
                    <span className="rule-id-badge">{ev.rule_id}</span>
                  </div>
                  <div className="det-meta">
                    <span>{ev.timestamp}s</span>
                    <span>{ev.is_violation ? 'violation' : 'match'}</span>
                  </div>
                </div>
              ))}
              {eventLog.length === 0 && (
                <p style={{ opacity: 0.4, textAlign: 'center', marginTop: '3rem', fontSize: '0.8rem' }}>
                  Events will appear here as rules fire.
                </p>
              )}
              <div ref={logEndRef} />
            </div>
          </div>

        </aside>
      </main>
    </div>
  );
}
