import { useState, useRef, useEffect, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:5050";

/* ───────── Lead metadata ───────── */
const LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];
const LEAD_COLORS = {
  I:"#0c8ce9", II:"#0d9488", III:"#8b5cf6",
  aVR:"#d97706", aVL:"#dc2626", aVF:"#059669",
  V1:"#2563eb", V2:"#0891b2", V3:"#7c3aed",
  V4:"#ea580c", V5:"#db2777", V6:"#65a30d",
};

/* Spatial groupings */
const SPATIAL_LEADS = {
  "RA (Right Arm)": ["I", "aVR"],
  "LA (Left Arm)":  ["II", "aVL"],
  "LL (Left Leg)":  ["III", "aVF"],
};

/* ───────── Model options ───────── */
const MODEL_OPTIONS = [
  {
    id: "hybrid",
    label: "Hybrid Ensemble",
    badge: "Recommended",
    auroc: 0.8707,
    auprc: 0.2589,
    tpr: 0.4958,
    description: "Dual-pathway: 2D contour + 1D signal + demographics fusion",
    needsDemographics: true,
    color: "brand",
    iconBg: "bg-pastel-blue",
    iconColor: "text-brand-500",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <rect x="3" y="3" width="7" height="7" rx="1.5" />
        <rect x="14" y="3" width="7" height="7" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" />
        <rect x="14" y="14" width="7" height="7" rx="1.5" />
        <path d="M10 6.5h4M10 17.5h4M6.5 10v4M17.5 10v4" strokeDasharray="2 2" />
      </svg>
    ),
  },
  {
    id: "2d",
    label: "2D Visual Model",
    badge: null,
    auroc: 0.7079,
    auprc: 0.0984,
    tpr: 0.2899,
    description: "ECG contour image analysis via Vision Transformer",
    needsDemographics: false,
    color: "teal",
    iconBg: "bg-pastel-mint",
    iconColor: "text-medical-teal",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <rect x="3" y="3" width="18" height="18" rx="3" />
        <path d="M3 15l4-4 3 3 4-5 7 6" />
      </svg>
    ),
  },
  {
    id: "1d",
    label: "1D Signal Model",
    badge: null,
    auroc: 0.8567,
    auprc: 0.2295,
    tpr: 0.4482,
    description: "Raw 12-lead ECG signal + patient demographics",
    needsDemographics: true,
    color: "purple",
    iconBg: "bg-pastel-lilac",
    iconColor: "text-purple-500",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <path d="M2 12h3l2-6 3 12 2-8 2 4 2-2h6" />
      </svg>
    ),
  },
];

/* ───────── Sample ECGs ───────── */
const SAMPLE_ECGS = [
  { dataset: "SAMITROP", label: "SAMITROP", desc: "Chagas endemic region (Brazil)", color: "brand", path: "samitrop", files: ["100726.hea", "100726.dat"] },
  { dataset: "PTB-XL",   label: "PTB-XL",   desc: "European clinical database",    color: "teal",   path: "ptbxl",   files: ["00001_lr.hea", "00001_lr.dat"] },
  { dataset: "CODE-15%", label: "CODE-15%", desc: "Brazilian 12-lead database",    color: "purple", path: "code15s", files: ["13.hea", "13.dat"] },
];

/* ───────── Preprocessing pipeline stages ───────── */
const STAGE_ORDER = ["raw", "baseline_removed", "sig_100hz", "sig_500hz"];
const STAGE_ICONS = {
  raw:              "⚡",
  baseline_removed: "🔧",
  sig_100hz:        "📉",
  sig_500hz:        "🎨",
};

/* ───────── Chagas disease stats ───────── */
const CHAGAS_STATS = [
  { value: "6–7M",  label: "People Infected",        icon: "🌍", color: "brand" },
  { value: "30%",   label: "Develop Cardiac Form",   icon: "❤️", color: "teal" },
  { value: "70+",   label: "Countries Affected",     icon: "🗺️", color: "purple" },
  { value: ">$7B",  label: "Annual Economic Burden",  icon: "💊", color: "orange" },
];

/* ═══════════════════════════════════════════════════
   SHARED COMPONENTS
════════════════════════════════════════════════════ */

/* ── Circular gauge ── */
function ProbGauge({ probability, isPositive }) {
  const r = 52;
  const circ = 2 * Math.PI * r;
  const offset = circ - probability * circ;
  const pct = (probability * 100).toFixed(1);
  return (
    <div className="flex flex-col items-center">
      <div className="relative w-32 h-32">
        <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
          <circle cx="60" cy="60" r={r} fill="none" stroke="#e2e8f0" strokeWidth="8" />
          <circle cx="60" cy="60" r={r} fill="none" strokeWidth="8" strokeLinecap="round"
            stroke={isPositive ? "url(#gaugeRed)" : "url(#gaugeGreen)"}
            strokeDasharray={circ} strokeDashoffset={offset} className="animate-gauge" />
          <defs>
            <linearGradient id="gaugeGreen" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#10b981" />
              <stop offset="100%" stopColor="#06d6a0" />
            </linearGradient>
            <linearGradient id="gaugeRed" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#ef4444" />
              <stop offset="100%" stopColor="#f97316" />
            </linearGradient>
          </defs>
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-fluid-3xl font-extrabold tracking-tight text-slate-800">{pct}</span>
          <span className="text-fluid-xs text-slate-700 font-semibold">%</span>
        </div>
      </div>
    </div>
  );
}

/* ── Animated ECG line ── */
function EcgLine({ className = "" }) {
  return (
    <svg viewBox="0 0 400 60" className={`h-8 ${className}`} preserveAspectRatio="none">
      <path d="M0,30 L60,30 L80,10 L100,50 L120,20 L140,40 L160,25 L180,30 L240,30 L260,5 L280,55 L300,15 L320,35 L340,30 L400,30"
        fill="none" stroke="url(#ecgGrad)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="ecg-line" />
      <defs>
        <linearGradient id="ecgGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#0c8ce9" />
          <stop offset="50%" stopColor="#10b981" />
          <stop offset="100%" stopColor="#0c8ce9" />
        </linearGradient>
      </defs>
    </svg>
  );
}

/* ── Blob ── */
function Blob({ className }) {
  return <div className={`absolute rounded-full blur-3xl pointer-events-none ${className}`} />;
}

/* ── Single-lead waveform (pure SVG) ── */
function WaveformChart({ signal, color = "#0c8ce9", height = 72, showGrid = true }) {
  if (!signal || signal.length === 0) return (
    <div className="flex items-center justify-center" style={{ height }}>
      <span className="text-fluid-xs text-slate-500">No data</span>
    </div>
  );
  const W = 600;
  const H = height;
  const pad = 4;
  const min = Math.min(...signal);
  const max = Math.max(...signal);
  const range = max - min || 1;
  const pts = signal.map((v, i) => {
    const x = (i / (signal.length - 1)) * (W - pad * 2) + pad;
    const y = H - pad - ((v - min) / range) * (H - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={`w-full ${showGrid ? "chart-grid" : ""}`}
      style={{ height }} preserveAspectRatio="none">
      {showGrid && (
        <line x1="0" y1={H / 2} x2={W} y2={H / 2} stroke="rgba(0,0,0,0.08)" strokeWidth="1" />
      )}
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

/* ── Multi-lead grid (responsive cols) ── */
function MultiLeadGrid({ signals, activeSub, selectedLeads, onToggleLead }) {
  if (!signals || signals.length === 0) return null;

  const leadsToShow = activeSub === "spatial"
    ? Object.values(SPATIAL_LEADS).flat()
    : LEAD_NAMES;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
      {leadsToShow.map((lead) => {
        const leadIdx = LEAD_NAMES.indexOf(lead);
        const sig = signals[leadIdx] ?? [];
        const active = selectedLeads.has(lead);
        return (
          <div key={lead}
            className={`rounded-xl border-2 transition-all duration-200 overflow-hidden cursor-pointer
              ${active ? "border-brand-200 shadow-sm" : "border-surface-200 opacity-40"}`}
            onClick={() => onToggleLead(lead)}>
            <div className="flex items-center justify-between px-3 pt-2.5 pb-1.5">
              <span className="lead-label text-fluid-xs font-bold" style={{ color: LEAD_COLORS[lead] }}>{lead}</span>
              {activeSub === "spatial" && (
                <span className="text-fluid-2xs text-slate-600">
                  {Object.entries(SPATIAL_LEADS).find(([, arr]) => arr.includes(lead))?.[0] ?? ""}
                </span>
              )}
              <span className={`w-2 h-2 rounded-full flex-shrink-0 ${active ? "opacity-100" : "opacity-25"}`}
                style={{ background: LEAD_COLORS[lead] }} />
            </div>
            <div className="px-1 pb-2">
              <WaveformChart signal={sig} color={active ? LEAD_COLORS[lead] : "#94a3b8"} height={60} showGrid={active} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ── Stage card for preprocessing tab ── */
function StageCard({ stageKey, data, isLast }) {
  const [expanded, setExpanded] = useState(false);
  if (!data) return null;
  const { label, description, color, signal, stats } = data;
  const leadsToShow = expanded ? LEAD_NAMES : LEAD_NAMES.slice(0, 3);

  return (
    <div className="relative">
      {!isLast && <div className="stage-connector h-6 my-1" />}
      <div className="card shadow-card overflow-hidden">
        <div className="flex items-center gap-3 p-4 border-b border-surface-100">
          <span className="text-2xl flex-shrink-0">{STAGE_ICONS[stageKey]}</span>
          <div className="flex-1 min-w-0">
            <div className="text-fluid-sm font-bold text-slate-800">{label}</div>
            <div className="text-fluid-xs text-slate-600 mt-0.5 leading-snug">{description}</div>
          </div>
          <div className="hidden sm:flex items-center gap-2 flex-shrink-0 flex-wrap justify-end">
            <span className="px-2 py-1 rounded-lg bg-surface-100 text-fluid-xs font-semibold text-slate-700">
              {stats.fs} Hz
            </span>
            <span className="px-2 py-1 rounded-lg bg-surface-100 text-fluid-xs font-semibold text-slate-700">
              {stats.n_samples} samples
            </span>
            <span className="px-2 py-1 rounded-lg bg-surface-100 text-fluid-xs font-semibold text-slate-700">
              {stats.duration_s}s
            </span>
          </div>
          <button onClick={() => setExpanded(e => !e)}
            className="w-8 h-8 rounded-lg bg-surface-50 border border-surface-200 flex items-center justify-center text-slate-600 hover:text-slate-800 transition-colors flex-shrink-0">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={`w-4 h-4 transition-transform ${expanded ? "rotate-180" : ""}`}>
              <path d="M6 9l6 6 6-6" />
            </svg>
          </button>
        </div>
        <div className="p-4">
          <div className={`grid gap-2 ${expanded ? "grid-cols-1 sm:grid-cols-2 xl:grid-cols-3" : "grid-cols-3"}`}>
            {leadsToShow.map((lead, i) => (
              <div key={lead} className="rounded-lg overflow-hidden bg-surface-50 border border-surface-100">
                <div className="px-2 pt-1.5">
                  <span className="text-fluid-xs font-bold" style={{ color }}>{lead}</span>
                </div>
                <div className="px-1 pb-1">
                  <WaveformChart signal={signal[i] ?? []} color={color} height={48} showGrid={false} />
                </div>
              </div>
            ))}
          </div>
          {!expanded && (
            <button onClick={() => setExpanded(true)}
              className="mt-3 w-full text-fluid-xs text-brand-600 hover:text-brand-700 font-semibold text-center py-1">
              Show all 12 leads ↓
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════
   MAIN APP
════════════════════════════════════════════════════ */
export default function App() {
  /* ── Core state ── */
  const [files, setFiles] = useState([]);
  const [modelType, setModelType] = useState("hybrid");
  const [age, setAge] = useState("");
  const [sex, setSex] = useState("unknown");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [apiOk, setApiOk] = useState(null);
  const [activeTab, setActiveTab] = useState("analyze");
  const [sampleLoading, setSampleLoading] = useState(null);

  /* ── Signal / preview state ── */
  const [previewData, setPreviewData] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState("");
  const [signalSubTab, setSignalSubTab] = useState("twelve_lead");
  const [selectedLeads, setSelectedLeads] = useState(new Set(LEAD_NAMES));

  const dropRef = useRef(null);
  const fileInputRef = useRef(null);
  const selectedModel = MODEL_OPTIONS.find((m) => m.id === modelType);

  /* ── Health check ── */
  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((r) => r.ok && setApiOk(true))
      .catch(() => setApiOk(false));
  }, []);

  /* ── Helpers ── */
  const hasWFDBPair = useCallback(() => {
    const names = files.map((f) => f.name.toLowerCase());
    return names.some((n) => n.endsWith(".hea")) &&
      (names.some((n) => n.endsWith(".dat")) || names.some((n) => n.endsWith(".mat")));
  }, [files]);

  const handleFiles = (newFiles) => {
    setResult(null); setError("");
    setPreviewData(null); setPreviewError("");
    setFiles((prev) => {
      const existing = new Set(prev.map((f) => f.name));
      return [...prev, ...newFiles.filter((f) => !existing.has(f.name))];
    });
  };

  const removeFile = (name) => setFiles((prev) => prev.filter((f) => f.name !== name));

  /* ── Safe JSON parse helper ── */
  const safeJson = async (res) => {
    const text = await res.text();
    try {
      return JSON.parse(text);
    } catch {
      // Server returned HTML (likely error page or not running)
      throw new Error(
        res.ok
          ? "API returned an unexpected response. Try restarting the backend."
          : `API error ${res.status} — make sure app.py is running on port 5050.`
      );
    }
  };

  /* ── Auto-fetch preview when valid pair detected ── */
  useEffect(() => {
    if (!hasWFDBPair() || files.length === 0) { setPreviewData(null); return; }
    let cancelled = false;
    const fetchPreview = async () => {
      setPreviewLoading(true); setPreviewError("");
      try {
        const fd = new FormData();
        files.forEach((f) => fd.append("files", f));
        const res = await fetch(`${API_BASE}/api/preview`, { method: "POST", body: fd });
        const data = await safeJson(res);
        if (!res.ok) throw new Error(data.error || "Preview failed");
        if (!cancelled) setPreviewData(data);
      } catch (err) {
        if (!cancelled) setPreviewError(err.message);
      } finally {
        if (!cancelled) setPreviewLoading(false);
      }
    };
    fetchPreview();
    return () => { cancelled = true; };
  }, [files, hasWFDBPair]);

  /* ── Drag & drop ── */
  const onDrag = (e) => { e.preventDefault(); e.stopPropagation(); };
  const onDragIn = (e) => { onDrag(e); setDragActive(true); };
  const onDragOut = (e) => { onDrag(e); setDragActive(false); };
  const onDrop = (e) => {
    onDrag(e); setDragActive(false);
    if (e.dataTransfer.files?.length) handleFiles(Array.from(e.dataTransfer.files));
  };
  const onDropZoneKey = (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInputRef.current?.click(); }
  };

  /* ── Sample loader ── */
  const loadSample = async (sample) => {
    setSampleLoading(sample.dataset);
    setResult(null); setError(""); setFiles([]);
    setPreviewData(null); setPreviewError("");
    try {
      const loaded = await Promise.all(
        sample.files.map(async (name) => {
          const res = await fetch(`/samples/${sample.path}/${name}`);
          if (!res.ok) throw new Error(`Could not fetch ${name}`);
          const blob = await res.blob();
          return new File([blob], name);
        })
      );
      handleFiles(loaded);
    } catch (e) {
      setError(e.message || "Failed to load sample ECG.");
    } finally {
      setSampleLoading(null);
    }
  };

  /* ── Run prediction ── */
  const runPrediction = async () => {
    if (!hasWFDBPair()) { setError("Upload a matching .hea + .dat (or .mat) pair."); return; }
    if (selectedModel.needsDemographics && age !== "") {
      const p = parseInt(age, 10);
      if (isNaN(p) || p < 0 || p > 120) { setError("Please enter a valid age (0–120)."); return; }
    }
    setLoading(true); setError(""); setResult(null);
    try {
      const fd = new FormData();
      files.forEach((f) => fd.append("files", f));
      fd.append("model_type", modelType);
      if (selectedModel.needsDemographics) { fd.append("age", age || "50"); fd.append("sex", sex); }
      const res = await fetch(`${API_BASE}/api/predict`, { method: "POST", body: fd });
      const data = await safeJson(res);
      if (!res.ok) throw new Error(data.error || "Prediction failed");
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  /* ── Lead toggle ── */
  const toggleLead = (lead) => {
    setSelectedLeads((prev) => {
      const next = new Set(prev);
      if (next.has(lead)) { if (next.size > 1) next.delete(lead); }
      else next.add(lead);
      return next;
    });
  };

  const isPositive = result?.prediction === 1;
  const pct = result ? (result.probability * 100).toFixed(1) : null;

  /* ── Tab definitions ── */
  const TABS = [
    { key: "analyze",    label: "Analyze",       icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4"><path d="M2 12h3l2-6 3 12 2-8 2 4 2-2h6" strokeLinecap="round" strokeLinejoin="round" /></svg> },
    { key: "signals",    label: "Signals",       icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4"><path d="M22 12h-4l-3 9L9 3l-3 9H2" strokeLinecap="round" strokeLinejoin="round" /></svg> },
    { key: "preprocess", label: "Preprocessing", icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" strokeLinecap="round" strokeLinejoin="round" /></svg> },
    { key: "about",      label: "About",         icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4"><circle cx="12" cy="12" r="10" /><path d="M12 8v4M12 16h.01" strokeLinecap="round" /></svg> },
  ];

  /* ── Reusable empty state ── */
  const NoDataState = ({ icon, title, sub, ctaLabel, ctaAction, accent = "brand" }) => (
    <div className="card shadow-card p-10 flex flex-col items-center text-center">
      <div className={`w-16 h-16 rounded-2xl ${accent === "teal" ? "bg-pastel-mint" : "bg-pastel-blue"} flex items-center justify-center mb-4`}>
        {icon}
      </div>
      <h3 className="text-fluid-lg font-bold text-slate-700 mb-2">{title}</h3>
      <p className="text-fluid-sm text-slate-600 mb-5">{sub}</p>
      <button onClick={ctaAction}
        className={`px-5 py-2.5 rounded-xl text-white font-bold text-fluid-sm transition-all hover:scale-[1.02]
          ${accent === "teal" ? "bg-gradient-to-r from-medical-teal to-medical-green shadow-glow-green" : "bg-gradient-to-r from-brand-500 to-brand-600 shadow-brand"}`}>
        {ctaLabel}
      </button>
    </div>
  );

  /* ══════════════════════════════════════════════════
     RENDER
  ══════════════════════════════════════════════════ */
  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <Blob className="w-[60vw] h-[60vw] max-w-[500px] max-h-[500px] bg-brand-200/40 -top-[12vw] -left-[12vw] animate-float" />
      <Blob className="w-[50vw] h-[50vw] max-w-[400px] max-h-[400px] bg-pastel-mint/50 bottom-20 -right-[10vw] animate-float-slow" />
      <Blob className="w-[40vw] h-[40vw] max-w-[300px] max-h-[300px] bg-pastel-lilac/40 top-1/2 left-1/3 animate-float" />

      {/* ── 95 vw on mobile, 80 vw on xl+ ── */}
      <div className="relative z-10 mx-auto px-3 sm:px-5 py-5"
        style={{ maxWidth: "min(95vw, 1400px)" }}>

        {/* ══════════ NAV BAR ══════════ */}
        <nav className="print:hidden card shadow-card px-4 sm:px-5 py-3 flex items-center justify-between gap-2 mb-5 animate-fade-in-up flex-wrap">
          <div className="flex items-center gap-3 flex-shrink-0">
            <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-gradient-to-br from-brand-400 to-brand-600 shadow-brand">
              <svg viewBox="0 0 24 24" fill="none" className="w-5 h-5 text-white">
                <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" fill="currentColor" />
              </svg>
            </div>
            <div>
              <h1 className="text-fluid-xl font-extrabold tracking-tight">
                <span className="bg-gradient-to-r from-brand-500 to-brand-700 bg-clip-text text-transparent">ChagaSight</span>
              </h1>
              <p className="text-fluid-2xs text-slate-600 font-semibold -mt-0.5">AI-Powered ECG Analysis</p>
            </div>
          </div>

          <div className="flex items-center bg-surface-100 rounded-xl p-1 gap-0.5 flex-wrap" role="tablist">
            {TABS.map((tab) => (
              <button key={tab.key} role="tab" aria-selected={activeTab === tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-fluid-xs font-bold transition-all duration-200
                  ${activeTab === tab.key ? "bg-white text-brand-600 shadow-sm" : "text-slate-600 hover:text-slate-900"}`}>
                {tab.icon}
                <span className="hidden sm:inline">{tab.label}</span>
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2 flex-shrink-0">
            {previewLoading && (
              <span className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-brand-50 border border-brand-100 text-fluid-xs font-bold text-brand-600">
                <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="31.4 31.4" strokeLinecap="round" />
                </svg>
                Processing…
              </span>
            )}
            {previewData && !previewLoading && (
              <span className="px-2 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-fluid-xs font-bold text-emerald-700">
                ✓ Signals ready
              </span>
            )}
            <div className="flex items-center gap-1.5 px-2 py-1.5 rounded-full bg-surface-50 border border-surface-200">
              <span className={`w-2 h-2 rounded-full flex-shrink-0 ${apiOk === true ? "bg-medical-green animate-pulse-soft" : apiOk === false ? "bg-medical-red" : "bg-medical-orange animate-pulse-soft"}`} />
              <span className="hidden sm:block text-fluid-xs text-slate-700 font-bold whitespace-nowrap">
                {apiOk === true ? "API Live" : apiOk === false ? "Offline" : "Checking…"}
              </span>
            </div>
          </div>
        </nav>

        {/* ══════════ ANALYZE TAB ══════════ */}
        {activeTab === "analyze" && (
          <div className="space-y-5 animate-fade-in-up">
            <div className="card shadow-card overflow-hidden">
              <div className="flex items-center gap-6 p-5">
                <div className="flex-1">
                  <h2 className="text-fluid-2xl font-extrabold text-slate-800 mb-1">ECG Analysis</h2>
                  <p className="text-fluid-sm text-slate-700">Upload a 12-lead ECG to screen for Chagas disease risk</p>
                </div>
                <EcgLine className="w-40 hidden md:block opacity-50" />
              </div>
            </div>

            {/* 3-column grid: model | input | result */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">

              {/* ── COL 1: Model selector ── */}
              <div className="print:hidden" role="region" aria-label="Model selection">
                <div className="card shadow-card p-4">
                  <h3 className="text-fluid-xs font-bold text-slate-700 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4 text-brand-500">
                      <path d="M4 6h16M4 12h16M4 18h16" />
                    </svg>
                    Select Model
                  </h3>
                  <div className="space-y-3" role="radiogroup">
                    {MODEL_OPTIONS.map((opt) => {
                      const active = modelType === opt.id;
                      return (
                        <button key={opt.id} role="radio" aria-checked={active}
                          onClick={() => { setModelType(opt.id); setResult(null); setError(""); }}
                          className={`relative w-full text-left rounded-xl p-3 border-2 transition-all duration-300 group
                            ${active ? "model-active bg-pastel-blue/40" : "border-surface-200 hover:border-brand-200 hover:bg-surface-50"}`}>
                          {opt.badge && (
                            <span className="absolute -top-2.5 right-3 rounded-full bg-gradient-to-r from-brand-500 to-brand-600 px-2 py-0.5 text-[10px] font-bold text-white shadow-brand">
                              {opt.badge}
                            </span>
                          )}
                          <div className="flex items-start gap-3">
                            <div className={`w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 transition-colors
                              ${active ? opt.iconBg : "bg-surface-100 group-hover:bg-surface-200"}`}>
                              <div className={active ? opt.iconColor : "text-slate-500"}>{opt.icon}</div>
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="text-fluid-sm font-bold text-slate-800">{opt.label}</div>
                              <div className="text-fluid-2xs text-slate-600 mt-0.5 leading-tight">{opt.description}</div>
                              <div className="mt-2 space-y-1.5">
                                {[["AUROC", opt.auroc, "brand"], ["AUPRC", opt.auprc, "teal"]].map(([lbl, val, col]) => (
                                  <div key={lbl}>
                                    <div className="flex justify-between text-fluid-2xs mb-0.5">
                                      <span className="text-slate-600 font-semibold">{lbl}</span>
                                      <span className="font-bold text-slate-700">{Number(val).toFixed(3)}</span>
                                    </div>
                                    <div className="h-1.5 rounded-full bg-surface-200">
                                      <div className={`h-full rounded-full bg-gradient-to-r ${col === "brand" ? "from-brand-400 to-brand-500" : "from-medical-teal to-medical-green"} transition-all duration-700`}
                                        style={{ width: `${Number(val) * 100}%` }} />
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* ── COL 2: Input data ── */}
              <div className="print:hidden space-y-4" role="region" aria-label="Input data">
                {selectedModel.needsDemographics && (
                  <div className="card shadow-card p-4 animate-scale-in">
                    <h3 className="text-fluid-xs font-bold text-slate-700 uppercase tracking-wider mb-3 flex items-center gap-2">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4 text-medical-teal">
                        <circle cx="12" cy="7" r="4" /><path d="M5.5 21a6.5 6.5 0 0113 0" />
                      </svg>
                      Patient Demographics
                    </h3>
                    <div className="space-y-3">
                      <div>
                        <label htmlFor="age-input" className="block text-fluid-xs font-bold text-slate-700 mb-1.5">
                          Age <span className="text-slate-500 font-normal">(years)</span>
                        </label>
                        <input id="age-input" type="number" min="0" max="120" placeholder="e.g. 45"
                          value={age} onChange={(e) => setAge(e.target.value)}
                          className="w-full rounded-xl bg-surface-50 border border-surface-200 px-3 py-2.5 text-fluid-sm text-slate-800 placeholder:text-slate-400 transition-all" />
                      </div>
                      <div>
                        <label htmlFor="sex-select" className="block text-fluid-xs font-bold text-slate-700 mb-1.5">Biological Sex</label>
                        <select id="sex-select" value={sex} onChange={(e) => setSex(e.target.value)}
                          className="w-full rounded-xl bg-surface-50 border border-surface-200 px-3 py-2.5 text-fluid-sm text-slate-800 transition-all appearance-none cursor-pointer">
                          <option value="unknown">Unknown</option>
                          <option value="male">Male</option>
                          <option value="female">Female</option>
                        </select>
                      </div>
                    </div>
                  </div>
                )}

                <div className="card shadow-card p-4">
                  <h3 className="text-fluid-xs font-bold text-slate-700 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4 text-brand-500">
                      <path d="M12 16V4m0 0l-4 4m4-4l4 4" /><path d="M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
                    </svg>
                    Upload ECG Recording
                  </h3>
                  <p className="text-fluid-xs text-slate-700 font-bold uppercase tracking-wider mb-2">Try a Sample</p>
                  <div className="grid grid-cols-1 gap-1.5 mb-4">
                    {SAMPLE_ECGS.map((s) => (
                      <button key={s.dataset} onClick={() => loadSample(s)} disabled={sampleLoading !== null}
                        className={`rounded-xl border-2 p-3 text-left transition-all duration-200 hover:shadow-sm
                          ${s.color === "brand" ? "border-brand-200 bg-pastel-blue/50 hover:border-brand-400" :
                            s.color === "teal"  ? "border-teal-200 bg-pastel-mint/50 hover:border-teal-400" :
                              "border-purple-200 bg-pastel-lilac/50 hover:border-purple-400"}
                          ${sampleLoading === s.dataset ? "opacity-60 cursor-wait" : "cursor-pointer"}`}>
                        <div className={`text-fluid-xs font-extrabold
                          ${s.color === "brand" ? "text-brand-700" : s.color === "teal" ? "text-medical-teal" : "text-purple-700"}`}>
                          {sampleLoading === s.dataset ? "Loading…" : s.label}
                        </div>
                        <div className="text-fluid-2xs text-slate-600 mt-0.5 leading-tight">{s.desc}</div>
                      </button>
                    ))}
                  </div>

                  <div ref={dropRef} role="button" tabIndex={0}
                    aria-label="Upload ECG files"
                    onDragEnter={onDragIn} onDragLeave={onDragOut} onDragOver={onDrag} onDrop={onDrop}
                    onClick={() => fileInputRef.current?.click()} onKeyDown={onDropZoneKey}
                    className={`rounded-xl border-2 border-dashed cursor-pointer transition-all duration-300 p-5 text-center group
                      ${dragActive ? "drop-active" : "border-surface-300 hover:border-brand-300 hover:bg-pastel-blue/30"}`}>
                    <input ref={fileInputRef} type="file" multiple accept=".hea,.dat,.mat"
                      onChange={(e) => handleFiles(Array.from(e.target.files))} className="hidden" />
                    <div className="w-12 h-12 mx-auto rounded-2xl bg-pastel-blue flex items-center justify-center mb-3 group-hover:scale-105 transition-transform">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6 text-brand-500">
                        <path d="M12 16V4m0 0l-4 4m4-4l4 4" strokeLinecap="round" strokeLinejoin="round" />
                        <path d="M2 17l.621 2.485A2 2 0 004.561 21h14.878a2 2 0 001.94-1.515L22 17" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </div>
                    <p className="text-fluid-sm text-slate-700 font-bold">Drag & drop WFDB files</p>
                    <p className="text-fluid-xs text-slate-600 mt-0.5">or <span className="text-brand-600 font-bold cursor-pointer hover:underline">browse</span></p>
                    <p className="text-fluid-2xs text-slate-500 mt-1">.hea + .dat or .mat pair required</p>
                  </div>

                  {files.length > 0 && (
                    <div className="mt-3 space-y-1.5">
                      {files.map((f) => (
                        <div key={f.name} className="flex items-center justify-between bg-surface-50 rounded-lg px-3 py-2 border border-surface-200">
                          <div className="flex items-center gap-2">
                            <div className="w-7 h-7 rounded-lg bg-pastel-blue flex items-center justify-center">
                              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5 text-brand-600">
                                <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z" /><path d="M14 2v6h6" />
                              </svg>
                            </div>
                            <div>
                              <span className="text-fluid-xs text-slate-800 font-semibold">{f.name}</span>
                              <span className="text-fluid-2xs text-slate-600 ml-1.5">({(f.size / 1024).toFixed(0)} KB)</span>
                            </div>
                          </div>
                          <button onClick={(e) => { e.stopPropagation(); removeFile(f.name); }}
                            className="w-8 h-8 rounded-lg hover:bg-red-50 flex items-center justify-center text-slate-500 hover:text-red-500 transition-all">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3.5 h-3.5">
                              <path d="M18 6L6 18M6 6l12 12" />
                            </svg>
                          </button>
                        </div>
                      ))}
                      <div className="flex items-center gap-2 mt-1 px-1">
                        {hasWFDBPair() ? (
                          <>
                            <span className="w-4 h-4 rounded-full bg-medical-green/10 flex items-center justify-center">
                              <svg viewBox="0 0 24 24" fill="currentColor" className="w-3 h-3 text-medical-green"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z" /></svg>
                            </span>
                            <span className="text-fluid-xs text-emerald-700 font-bold">Valid WFDB pair detected</span>
                          </>
                        ) : (
                          <>
                            <span className="w-4 h-4 rounded-full bg-medical-orange/10 flex items-center justify-center">
                              <svg viewBox="0 0 24 24" fill="currentColor" className="w-3 h-3 text-medical-orange"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z" /></svg>
                            </span>
                            <span className="text-fluid-xs text-amber-700 font-bold">Need .hea + .dat/.mat pair</span>
                          </>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* ── COL 3: Predict + Result ── */}
              <div className="space-y-4" role="region" aria-label="Prediction result">
                <button onClick={runPrediction} disabled={loading || files.length === 0}
                  className="print:hidden w-full relative overflow-hidden rounded-xl py-3.5 font-bold text-white text-fluid-sm tracking-wide transition-all duration-300
                    bg-gradient-to-r from-brand-500 via-brand-600 to-brand-700
                    hover:shadow-brand-lg hover:scale-[1.01]
                    disabled:opacity-40 disabled:hover:shadow-none disabled:hover:scale-100 disabled:cursor-not-allowed active:scale-[0.99]">
                  {!loading && <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/15 to-transparent animate-shimmer bg-[length:200%_100%]" />}
                  <span className="relative flex items-center justify-center gap-2">
                    {loading ? (
                      <><svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                        <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="31.4 31.4" strokeLinecap="round" />
                      </svg>Running Inference…</>
                    ) : (
                      <><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
                        <path d="M2 12h3l2-6 3 12 2-8 2 4 2-2h6" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>Analyze ECG for Chagas Risk</>
                    )}
                  </span>
                </button>

                {previewData && !loading && (
                  <button onClick={() => setActiveTab("signals")}
                    className="print:hidden w-full rounded-xl border-2 border-brand-200 bg-pastel-blue/40 py-2.5 text-fluid-sm font-bold text-brand-700 hover:border-brand-400 transition-all duration-200 flex items-center justify-center gap-2">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4">
                      <path d="M22 12h-4l-3 9L9 3l-3 9H2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    View ECG Signals →
                  </button>
                )}

                {error && (
                  <div className="rounded-xl bg-red-50 border border-red-200 p-3 flex items-start gap-2 animate-scale-in">
                    <div className="w-7 h-7 rounded-lg bg-red-100 flex items-center justify-center flex-shrink-0">
                      <svg viewBox="0 0 24 24" fill="currentColor" className="w-3.5 h-3.5 text-red-600">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
                      </svg>
                    </div>
                    <p className="text-fluid-sm text-red-700 font-semibold pt-0.5">{error}</p>
                  </div>
                )}

                {result && (
                  <div aria-live="polite" aria-atomic="true"
                    className={`card shadow-elevated p-5 space-y-4 animate-slide-up border-2
                    ${isPositive ? "border-red-200 bg-red-50/50" : "border-emerald-200 bg-emerald-50/50"}`}>
                    <h1 className="hidden print:block text-fluid-2xl font-extrabold text-slate-800 border-b border-surface-200 pb-3 mb-4 mt-2">
                      Chagas Disease Screening Report
                    </h1>
                    {selectedModel.needsDemographics && (
                      <div className="hidden print:flex items-center gap-8 mb-4 pb-4 border-b border-surface-100">
                        <div>
                          <div className="text-fluid-2xs text-slate-600 uppercase tracking-wider font-bold mb-1">Patient Age</div>
                          <div className="text-fluid-sm text-slate-800 font-semibold">{age ? `${age} years` : "Not specified"}</div>
                        </div>
                        <div>
                          <div className="text-fluid-2xs text-slate-600 uppercase tracking-wider font-bold mb-1">Biological Sex</div>
                          <div className="text-fluid-sm text-slate-800 font-semibold capitalize">{sex}</div>
                        </div>
                      </div>
                    )}
                    <div className="flex items-center justify-between">
                      <h2 className="text-fluid-lg font-extrabold text-slate-800">Prediction Result</h2>
                      <span className={`rounded-full px-3 py-1 text-fluid-xs font-bold tracking-wide shadow-sm
                        ${isPositive ? "bg-gradient-to-r from-red-500 to-orange-500 text-white shadow-glow-red"
                          : "bg-gradient-to-r from-emerald-500 to-teal-400 text-white shadow-glow-green"}`}>
                        {isPositive ? "⚠ POSITIVE" : "✓ NEGATIVE"}
                      </span>
                    </div>
                    <div className="flex flex-col items-center py-2">
                      <ProbGauge probability={result.probability} isPositive={isPositive} />
                      <p className={`mt-2 text-fluid-sm font-bold ${isPositive ? "text-red-600" : "text-emerald-700"}`}>
                        {result.interpretation}
                      </p>
                    </div>
                    <div className="bg-white rounded-xl p-3 border border-surface-200">
                      <div className="flex justify-between text-fluid-xs mb-1.5">
                        <span className="text-slate-700 font-bold">Chagas Probability</span>
                        <span className="font-extrabold text-slate-800">{pct}%</span>
                      </div>
                      <div className="h-2.5 rounded-full bg-surface-200 overflow-hidden">
                        <div className={`h-full rounded-full transition-all duration-1000 ${isPositive ? "bg-gradient-to-r from-red-400 to-orange-400" : "bg-gradient-to-r from-emerald-400 to-teal-400"}`}
                          style={{ width: `${pct}%` }} />
                      </div>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      {[
                        { label: "Record",    value: result.record },
                        { label: "Model",     value: <span className="capitalize">{result.model_type}{result.folds_used > 1 && <span className="text-slate-500"> ({result.folds_used}F)</span>}</span> },
                        { label: "Threshold", value: result.threshold },
                      ].map((item, i) => (
                        <div key={i} className="bg-white rounded-lg border border-surface-200 p-2.5 text-center">
                          <div className="text-fluid-2xs text-slate-600 uppercase tracking-wider font-bold">{item.label}</div>
                          <div className="text-fluid-xs text-slate-800 font-bold mt-0.5">{item.value}</div>
                        </div>
                      ))}
                    </div>
                    <button onClick={() => window.print()}
                      className="w-full flex items-center justify-center gap-2 rounded-xl border border-surface-200 bg-white py-2.5 font-bold text-slate-700 hover:border-brand-200 hover:bg-brand-50 hover:text-brand-700 transition-all duration-300 shadow-sm print:hidden group text-fluid-sm">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4 group-hover:-translate-y-0.5 transition-transform">
                        <path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1" strokeLinecap="round" strokeLinejoin="round" />
                        <path d="M12 15V3" strokeLinecap="round" strokeLinejoin="round" />
                        <path d="M8 11l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                      Download PDF Report
                    </button>
                  </div>
                )}

                {!result && !error && !loading && (
                  <div className="card shadow-card p-6 flex flex-col items-center text-center">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="w-12 h-12 mb-3 text-slate-300">
                      <path d="M22 12h-4l-3 9L9 3l-3 9H2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    <p className="text-fluid-sm font-bold text-slate-600">Upload an ECG and run analysis</p>
                    <p className="text-fluid-xs text-slate-500 mt-1">Result will appear here</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ══════════ SIGNALS TAB ══════════ */}
        {activeTab === "signals" && (
          <div className="space-y-5 animate-fade-in-up">
            <div className="card shadow-card p-5 flex items-center gap-4 flex-wrap">
              <div className="flex-1 min-w-0">
                <h2 className="text-fluid-2xl font-extrabold text-slate-800 mb-1">ECG Signal Viewer</h2>
                <p className="text-fluid-sm text-slate-700">Explore the raw ECG signal across all leads and spatial groupings</p>
              </div>
              {previewData && (
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="px-2 py-1 rounded-lg bg-surface-100 text-fluid-xs font-bold text-slate-700">{previewData.record}</span>
                  <span className="px-2 py-1 rounded-lg bg-pastel-blue text-fluid-xs font-bold text-brand-700">{previewData.original_fs} Hz</span>
                </div>
              )}
            </div>

            {!previewData && !previewLoading && !previewError && (
              <NoDataState
                icon={<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-8 h-8 text-brand-500"><path d="M22 12h-4l-3 9L9 3l-3 9H2" strokeLinecap="round" strokeLinejoin="round" /></svg>}
                title="No ECG loaded"
                sub="Upload or load a sample ECG in the Analyze tab first"
                ctaLabel="← Go to Analyze"
                ctaAction={() => setActiveTab("analyze")}
              />
            )}

            {previewLoading && (
              <div className="card shadow-card p-10 flex flex-col items-center">
                <svg className="w-10 h-10 animate-spin text-brand-500 mb-4" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="31.4 31.4" strokeLinecap="round" />
                </svg>
                <p className="text-fluid-base font-bold text-slate-700">Processing ECG pipeline…</p>
                <p className="text-fluid-sm text-slate-600 mt-1">Reading signal and running preprocessing</p>
              </div>
            )}

            {previewError && (
              <div className="rounded-xl bg-red-50 border border-red-200 p-5 flex flex-col gap-2">
                <div className="flex items-start gap-3">
                  <svg viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
                  </svg>
                  <div>
                    <p className="text-fluid-sm text-red-700 font-bold">Preview failed</p>
                    <p className="text-fluid-xs text-red-600 mt-0.5">{previewError}</p>
                  </div>
                </div>
                <button onClick={() => setActiveTab("analyze")}
                  className="self-start mt-1 px-4 py-2 rounded-xl bg-white border border-red-200 text-fluid-xs font-bold text-red-600 hover:bg-red-50 transition-colors">
                  ← Back to Analyze
                </button>
              </div>
            )}

            {previewData && !previewLoading && (
              <>
                {/* Sub-tab selector */}
                <div className="flex items-center gap-1 p-1 bg-surface-100 rounded-xl w-fit flex-wrap">
                  {[
                    { key: "twelve_lead", label: "12-Lead" },
                    { key: "temporal",    label: "Temporal (1D, 100 Hz)" },
                    { key: "spatial",     label: "Spatial (RA · LA · LL)" },
                  ].map((st) => (
                    <button key={st.key} onClick={() => setSignalSubTab(st.key)}
                      className={`px-4 py-2 rounded-lg text-fluid-xs font-bold transition-all duration-200
                        ${signalSubTab === st.key ? "sub-tab-active" : "text-slate-600 hover:text-slate-900"}`}>
                      {st.label}
                    </button>
                  ))}
                </div>

                {/* Lead toggles */}
                <div className="card shadow-card p-4">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-fluid-xs font-bold text-slate-700 uppercase tracking-wider">Lead Visibility</span>
                    <div className="flex gap-2">
                      <button onClick={() => setSelectedLeads(new Set(LEAD_NAMES))}
                        className="text-fluid-xs px-2.5 py-1 rounded-lg bg-surface-100 text-slate-700 hover:bg-surface-200 font-bold transition">All</button>
                      <button onClick={() => setSelectedLeads(new Set([LEAD_NAMES[0]]))}
                        className="text-fluid-xs px-2.5 py-1 rounded-lg bg-surface-100 text-slate-700 hover:bg-surface-200 font-bold transition">None</button>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {LEAD_NAMES.map((lead) => (
                      <button key={lead} onClick={() => toggleLead(lead)}
                        className={`px-2.5 py-1 rounded-lg text-fluid-xs font-bold transition-all duration-200 border-2
                          ${selectedLeads.has(lead) ? "text-white border-transparent shadow-sm" : "text-slate-600 border-surface-200 bg-white"}`}
                        style={selectedLeads.has(lead) ? { backgroundColor: LEAD_COLORS[lead], borderColor: LEAD_COLORS[lead] } : {}}>
                        {lead}
                      </button>
                    ))}
                  </div>
                </div>

                {/* 12-lead / Temporal view */}
                {(signalSubTab === "twelve_lead" || signalSubTab === "temporal") && (
                  <div className="card shadow-card p-4">
                    <h3 className="text-fluid-sm font-bold text-slate-800 mb-1">
                      {signalSubTab === "twelve_lead" ? "12-Lead ECG" : "Temporal Signal — 1D Pathway Input (100 Hz, 1000 samples)"}
                    </h3>
                    <p className="text-fluid-xs text-slate-600 mb-3">
                      {signalSubTab === "twelve_lead"
                        ? `Raw WFDB signal · ${previewData.stages.raw.stats.fs} Hz · ${previewData.stages.raw.stats.n_samples} samples · ${previewData.stages.raw.stats.duration_s}s`
                        : "After resampling to 100 Hz, per-lead z-score normalisation and pad/trim to 1 000 samples"}
                    </p>
                    <MultiLeadGrid
                      signals={signalSubTab === "twelve_lead"
                        ? previewData.stages.raw.signal
                        : previewData.stages.sig_100hz.signal}
                      activeSub={signalSubTab}
                      selectedLeads={selectedLeads}
                      onToggleLead={toggleLead}
                    />
                  </div>
                )}

                {/* Spatial view */}
                {signalSubTab === "spatial" && (
                  <div className="space-y-4">
                    {Object.entries(SPATIAL_LEADS).map(([group, leads]) => (
                      <div key={group} className="card shadow-card p-4">
                        <h3 className="text-fluid-sm font-bold text-slate-800 mb-3">{group}</h3>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                          {leads.map((lead) => {
                            const idx = LEAD_NAMES.indexOf(lead);
                            const sig = previewData.stages.raw.signal[idx] ?? [];
                            const active = selectedLeads.has(lead);
                            return (
                              <div key={lead}
                                className={`rounded-xl border-2 overflow-hidden transition-all cursor-pointer
                                  ${active ? "border-brand-200" : "border-surface-200 opacity-40"}`}
                                onClick={() => toggleLead(lead)}>
                                <div className="flex items-center justify-between px-3 pt-2.5 pb-1.5">
                                  <span className="text-fluid-sm font-bold" style={{ color: LEAD_COLORS[lead] }}>{lead}</span>
                                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: LEAD_COLORS[lead] }} />
                                </div>
                                <div className="px-1 pb-2">
                                  <WaveformChart signal={sig} color={active ? LEAD_COLORS[lead] : "#94a3b8"} height={72} showGrid={active} />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* 2D Contour image */}
                <div className="card shadow-card p-4">
                  <h3 className="text-fluid-sm font-bold text-slate-800 mb-1">
                    2D Contour Image <span className="text-slate-600 font-normal">(2D ViT input)</span>
                  </h3>
                  <p className="text-fluid-xs text-slate-700 mb-3">
                    Created via Wilson Central Terminal (WCT) re-referencing to simulate 3 electrode perspectives. 
                    Each channel represents the signal from the perspective of a different reference limb.
                  </p>
                  
                  {previewData.contours ? (
                    <div className="space-y-4">
                      {["RA", "LA", "LL"].map((ref) => (
                        <div key={ref}>
                          <h4 className="text-fluid-xs font-bold text-slate-700 mb-1">{ref}-Referenced Channel</h4>
                          <div className="overflow-x-auto rounded-xl border border-surface-200 bg-black">
                            <img src={previewData.contours[ref]} alt={`2D ECG contour ${ref}`}
                              className="w-full" style={{ imageRendering: "pixelated", minWidth: "600px" }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="overflow-x-auto rounded-xl border border-surface-200 bg-black">
                      <img src={previewData.contour_image} alt="2D ECG contour combined"
                        className="w-full" style={{ imageRendering: "pixelated", minWidth: "600px" }} />
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        )}

        {/* ══════════ PREPROCESSING TAB ══════════ */}
        {activeTab === "preprocess" && (
          <div className="space-y-5 animate-fade-in-up">
            <div className="card shadow-card p-5">
              <h2 className="text-fluid-2xl font-extrabold text-slate-800 mb-1">Preprocessing Pipeline</h2>
              <p className="text-fluid-sm text-slate-700">
                See exactly how the raw WFDB signal is transformed at each stage before reaching the AI models.
              </p>
            </div>

            {!previewData && !previewLoading && !previewError && (
              <NoDataState
                icon={<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-8 h-8 text-medical-teal"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" strokeLinecap="round" strokeLinejoin="round" /></svg>}
                title="No ECG loaded"
                sub="Upload an ECG in the Analyze tab to see the preprocessing steps"
                ctaLabel="← Go to Analyze"
                ctaAction={() => setActiveTab("analyze")}
                accent="teal"
              />
            )}

            {previewLoading && (
              <div className="card shadow-card p-10 flex flex-col items-center">
                <svg className="w-10 h-10 animate-spin text-medical-teal mb-4" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="31.4 31.4" strokeLinecap="round" />
                </svg>
                <p className="text-fluid-base font-bold text-slate-700">Running preprocessing pipeline…</p>
              </div>
            )}

            {previewError && (
              <div className="rounded-xl bg-red-50 border border-red-200 p-4 text-fluid-sm text-red-700 font-semibold">{previewError}</div>
            )}

            {previewData && !previewLoading && (
              <div className="space-y-2">
                {/* Pipeline banner */}
                <div className="rounded-xl bg-gradient-to-r from-brand-50 to-pastel-mint border border-brand-100 p-4 flex flex-wrap gap-3 items-center">
                  {STAGE_ORDER.map((key, i) => {
                    const s = previewData.stages[key];
                    return (
                      <div key={key} className="flex items-center gap-2">
                        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white border border-surface-200 shadow-sm">
                          <span className="text-base">{STAGE_ICONS[key]}</span>
                          <span className="text-fluid-xs font-bold text-slate-700">{s?.label ?? key}</span>
                        </div>
                        {i < STAGE_ORDER.length - 1 && (
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4 text-slate-400">
                            <path d="M9 18l6-6-6-6" />
                          </svg>
                        )}
                      </div>
                    );
                  })}
                  <div className="flex items-center gap-2 ml-auto">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4 text-slate-400">
                      <path d="M9 18l6-6-6-6" />
                    </svg>
                    <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-gradient-to-r from-brand-500 to-brand-600 text-white shadow-brand">
                      <span className="text-base">🧠</span>
                      <span className="text-fluid-xs font-bold">ViT Model</span>
                    </div>
                  </div>
                </div>

                {STAGE_ORDER.map((key, i) => (
                  <StageCard key={key} stageKey={key} data={previewData.stages[key]}
                    isLast={i === STAGE_ORDER.length - 1} />
                ))}

                {/* 2D image stage */}
                <div className="relative">
                  <div className="stage-connector h-6 my-1" />
                  <div className="card shadow-card overflow-hidden">
                    <div className="flex items-center gap-3 p-4 border-b border-surface-100">
                      <span className="text-2xl">🖼️</span>
                      <div className="flex-1">
                        <div className="text-fluid-sm font-bold text-slate-800">2D Contour Image — ViT Input</div>
                        <div className="text-fluid-xs text-slate-600 mt-0.5">
                          WCT re-referencing creates three electrode perspectives (RA · LA · LL). Stacked to shape: 3 × 24 × 2 048 uint8.
                        </div>
                      </div>
                      <div className="hidden sm:flex flex-col gap-1 items-end flex-shrink-0">
                        <span className="px-2 py-1 rounded-lg bg-surface-100 text-fluid-xs font-bold text-slate-700">3 × 24 × 2048</span>
                        <span className="px-2 py-1 rounded-lg bg-surface-100 text-fluid-xs font-bold text-slate-700">uint8</span>
                      </div>
                    </div>
                    
                    {previewData.contours ? (
                      <div className="p-4 space-y-4">
                        {["RA", "LA", "LL"].map(ref => (
                           <div key={ref}>
                             <div className="text-fluid-xs font-bold text-slate-600 mb-1">{ref} Channel</div>
                             <div className="overflow-x-auto rounded-xl border border-surface-200 bg-black">
                               <img src={previewData.contours[ref]} alt={`2D ECG contour ${ref}`} className="w-full"
                                 style={{ imageRendering: "pixelated", minWidth: "500px" }} />
                             </div>
                           </div>
                        ))}
                      </div>
                    ) : (
                      <div className="p-4">
                        <div className="overflow-x-auto rounded-xl border border-surface-200 bg-black">
                          <img src={previewData.contour_image} alt="2D ECG contour combined" className="w-full"
                            style={{ imageRendering: "pixelated", minWidth: "500px" }} />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ══════════ ABOUT TAB ══════════ */}
        {activeTab === "about" && (
          <div className="space-y-6 animate-fade-in-up">

            {/* Hero */}
            <div className="card shadow-card overflow-hidden">
              <div className="flex flex-col md:flex-row items-center gap-6 p-7">
                <div className="flex-1">
                  <h2 className="text-fluid-3xl font-extrabold text-slate-800 mb-3">
                    Screening Chagas Disease
                    <br />
                    <span className="bg-gradient-to-r from-brand-500 to-medical-teal bg-clip-text text-transparent">
                      from the ECG
                    </span>
                  </h2>
                  <p className="text-fluid-base text-slate-700 leading-relaxed mb-4">
                    ChagaSight is an AI screening tool that detects signs of Chagas cardiomyopathy
                    directly from a standard 12-lead ECG. A dual-pathway Vision Transformer analyses
                    the ECG as both a 2D contour image and a 1D signal. A 5-fold ensemble delivers
                    a robust Chagas risk probability.
                  </p>
                  <div className="flex flex-wrap gap-2 text-fluid-xs">
                    <span className="px-2.5 py-1 bg-pastel-blue rounded-lg font-bold text-brand-700">Final Year Project</span>
                    <span className="px-2.5 py-1 bg-pastel-mint rounded-lg font-bold text-medical-teal">173M Parameters</span>
                    <span className="px-2.5 py-1 bg-pastel-lilac rounded-lg font-bold text-purple-700">5-Fold CV</span>
                    <span className="px-2.5 py-1 bg-surface-100 rounded-lg font-bold text-slate-700">366 181 ECGs</span>
                  </div>
                </div>
                <div className="w-full md:w-64 flex-shrink-0">
                  <img src="/medical_hero.png" alt="Medical AI illustration" className="w-full rounded-xl shadow-card" />
                </div>
              </div>
            </div>

            {/* Chagas Disease */}
            <div className="card shadow-card p-6">
              <h3 className="text-fluid-xl font-extrabold text-slate-800 mb-2 flex items-center gap-2">
                <span className="text-2xl">🦠</span> What is Chagas Disease?
              </h3>
              <p className="text-fluid-sm text-slate-700 mb-5 leading-relaxed">
                Chagas disease (American trypanosomiasis) is a chronic, life-threatening illness caused by the
                protozoan parasite <em>Trypanosoma cruzi</em>. Endemic primarily in Latin America, it is transmitted
                mainly through triatomine ("kissing") bugs. Up to 30% of chronically infected individuals develop
                severe cardiac complications — often decades after the initial infection during a silent phase.
              </p>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-5">
                {CHAGAS_STATS.map((s) => (
                  <div key={s.label} className={`rounded-xl p-4 text-center
                    ${s.color === "brand" ? "bg-pastel-blue border border-brand-100" :
                      s.color === "teal" ? "bg-pastel-mint border border-teal-100" :
                      s.color === "purple" ? "bg-pastel-lilac border border-purple-100" :
                      "bg-pastel-peach border border-orange-100"}`}>
                    <div className="text-2xl mb-1">{s.icon}</div>
                    <div className={`text-fluid-2xl font-extrabold
                      ${s.color === "brand" ? "text-brand-700" : s.color === "teal" ? "text-medical-teal" : s.color === "purple" ? "text-purple-700" : "text-orange-600"}`}>
                      {s.value}
                    </div>
                    <div className="text-fluid-xs text-slate-700 font-bold mt-0.5">{s.label}</div>
                  </div>
                ))}
              </div>
              <div className="rounded-xl bg-surface-50 border border-surface-200 p-4">
                <h4 className="text-fluid-sm font-bold text-slate-800 mb-2">Cardiac Manifestations &amp; ECG Findings</h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {[
                    { title: "Right Bundle Branch Block (RBBB)", desc: "Most common — present in ~20% of chronic patients" },
                    { title: "Left Anterior Fascicular Block",   desc: "Often combined with RBBB (bifascicular block)" },
                    { title: "QRS Prolongation",                 desc: "Progressive ventricular conduction delay" },
                    { title: "ST-T Wave Abnormalities",          desc: "Myocardial injury pattern from fibrosis" },
                    { title: "Ventricular Arrhythmias",          desc: "PVCs and VT from scar-mediated re-entry" },
                    { title: "Low Voltage / Q Waves",            desc: "Extensive myocardial replacement by fibrosis" },
                  ].map((item) => (
                    <div key={item.title} className="flex items-start gap-2 p-2.5 rounded-lg bg-white border border-surface-100">
                      <div className="w-1.5 h-1.5 rounded-full bg-brand-500 mt-2 flex-shrink-0" />
                      <div>
                        <div className="text-fluid-xs font-bold text-slate-800">{item.title}</div>
                        <div className="text-fluid-xs text-slate-600 mt-0.5">{item.desc}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="mt-4 rounded-xl bg-amber-50 border border-amber-200 p-3.5 text-fluid-sm text-amber-800 leading-relaxed">
                <span className="font-bold">Why ECG Screening?</span> The 12-lead ECG is inexpensive, non-invasive,
                and widely available in endemic regions. Early detection enables timely intervention — anti-parasitic
                treatment is most effective early, and cardiac monitoring can prevent sudden death from arrhythmia.
              </div>
            </div>

            {/* Performance metrics */}
            <div className="card shadow-card p-6">
              <h3 className="text-fluid-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
                <span className="w-8 h-8 rounded-lg bg-pastel-blue flex items-center justify-center text-brand-600">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4">
                    <path d="M3 17l4-8 4 4 4-6 4 4" /><path d="M3 21h18" />
                  </svg>
                </span>
                Validated Performance
              </h3>
              <div className="overflow-x-auto rounded-xl border border-surface-200">
                <table className="w-full text-fluid-xs">
                  <thead>
                    <tr className="bg-surface-50 border-b border-surface-200">
                      <th className="text-left px-4 py-3 font-bold text-slate-700 uppercase tracking-wider">Model</th>
                      <th className="px-4 py-3 font-bold text-slate-700 uppercase tracking-wider text-center">AUROC</th>
                      <th className="px-4 py-3 font-bold text-slate-700 uppercase tracking-wider text-center">AUPRC</th>
                      <th className="px-4 py-3 font-bold text-slate-700 uppercase tracking-wider text-center">TPR@5%FPR</th>
                      <th className="px-4 py-3 font-bold text-slate-700 uppercase tracking-wider text-center">Parameters</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { model: "Hybrid Ensemble ⭐", auroc: "0.8707 [0.8665–0.8746]", auprc: "0.2589 [0.2489–0.2685]", tpr: "49.6%", params: "~173M + ensemble" },
                      { model: "1D Signal Model",   auroc: "0.8567",                  auprc: "0.2295",                 tpr: "44.8%", params: "~87M" },
                      { model: "2D Visual Model",   auroc: "0.7079",                  auprc: "0.0984",                 tpr: "29.0%", params: "~86M" },
                    ].map((row, i) => (
                      <tr key={i} className={`border-b border-surface-100 ${i === 0 ? "bg-brand-50/40" : ""}`}>
                        <td className="px-4 py-3 font-bold text-slate-800">{row.model}</td>
                        <td className="px-4 py-3 text-center font-bold text-slate-700">{row.auroc}</td>
                        <td className="px-4 py-3 text-center text-slate-700">{row.auprc}</td>
                        <td className="px-4 py-3 text-center text-slate-700">{row.tpr}</td>
                        <td className="px-4 py-3 text-center text-slate-600">{row.params}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-fluid-xs text-slate-600 mt-3">
                5-fold cross-validation · 366,181 ECGs (8,190 positive) · Bootstrap CI (1,000 resamples)
              </p>
            </div>

            {/* How it works */}
            <div className="card shadow-card p-6">
              <h3 className="text-fluid-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                <span className="w-8 h-8 rounded-lg bg-pastel-blue flex items-center justify-center text-brand-600">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4">
                    <circle cx="12" cy="12" r="10" /><path d="M12 8v4l2 2" />
                  </svg>
                </span>
                How It Works
              </h3>
              <div className="space-y-3">
                {[
                  { step: "1", text: "Upload a 12-lead ECG recording (.hea + .dat WFDB files)" },
                  { step: "2", text: "Baseline removal (bandpass 0.5–40 Hz) cleans the raw signal" },
                  { step: "3", text: "2D path: resampled to 500 Hz → WCT re-referencing → 3 × 24 × 2 048 contour image" },
                  { step: "4", text: "1D path: resampled to 100 Hz → z-score normalised → 12 × 1 000 tensor + demographics" },
                  { step: "5", text: "Both pathways run through 12-layer AOL Vision Transformers in parallel" },
                  { step: "6", text: "Outputs fused → 5-fold ensemble averages → Chagas risk probability" },
                ].map((s) => (
                  <div key={s.step} className="flex items-start gap-3">
                    <span className="w-7 h-7 flex-shrink-0 rounded-full bg-gradient-to-br from-brand-400 to-brand-600 text-white text-fluid-xs font-bold flex items-center justify-center shadow-brand">
                      {s.step}
                    </span>
                    <span className="text-fluid-sm text-slate-700 pt-0.5">{s.text}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Disclaimer */}
            <div className="rounded-xl bg-amber-50 border border-amber-200 p-4 text-fluid-sm text-amber-800 leading-relaxed">
              <span className="font-bold">Research Prototype.</span> ChagaSight is an academic Final Year Project
              and is not approved for clinical use. Results should not be used to make medical decisions.
              Always consult a qualified healthcare professional.
            </div>

            <button onClick={() => setActiveTab("analyze")}
              className="w-full card shadow-card p-4 text-center hover:shadow-card-hover transition-all duration-300 group">
              <span className="text-fluid-sm font-bold text-brand-600 group-hover:text-brand-700 transition-colors">
                ← Start Analyzing ECGs
              </span>
            </button>
          </div>
        )}

        {/* Footer */}
        <footer className="print:hidden text-center text-fluid-xs text-slate-600 pt-8 pb-6 space-y-1">
          <p className="font-bold">ChagaSight v1.0 · Research Prototype</p>
          <p>Not validated for clinical use. Consult a medical professional for diagnosis.</p>
        </footer>
      </div>
    </div>
  );
}
