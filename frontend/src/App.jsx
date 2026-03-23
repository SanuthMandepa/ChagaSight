import { useState, useRef, useEffect } from "react";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:5050";

/* ───────── Model options ───────── */
const MODEL_OPTIONS = [
  {
    id: "hybrid",
    label: "Hybrid Ensemble",
    badge: "Recommended",
    auroc: 0.896,
    tpr: 0.504,
    description: "Dual-pathway: 2D contour + 1D signal + demographics fusion",
    needsDemographics: true,
    color: "brand",
    iconBg: "bg-pastel-blue",
    iconColor: "text-brand-500",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-7 h-7">
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
    auroc: 0.844,
    tpr: 0.463,
    description: "ECG contour image analysis via Vision Transformer",
    needsDemographics: false,
    color: "teal",
    iconBg: "bg-pastel-mint",
    iconColor: "text-medical-teal",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-7 h-7">
        <rect x="3" y="3" width="18" height="18" rx="3" />
        <path d="M3 15l4-4 3 3 4-5 7 6" />
      </svg>
    ),
  },
  {
    id: "1d",
    label: "1D Signal Model",
    badge: null,
    auroc: 0.828,
    tpr: 0.429,
    description: "Raw 12-lead ECG signal + patient demographics",
    needsDemographics: true,
    color: "purple",
    iconBg: "bg-pastel-lilac",
    iconColor: "text-purple-500",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-7 h-7">
        <path d="M2 12h3l2-6 3 12 2-8 2 4 2-2h6" />
      </svg>
    ),
  },
];

/* ───────── Sample ECGs ───────── */
const SAMPLE_ECGS = [
  { dataset: "SAMITROP", label: "SAMITROP", desc: "Chagas endemic region (Brazil)", color: "brand", path: "samitrop", files: ["100726.hea", "100726.dat"] },
  { dataset: "PTB-XL", label: "PTB-XL", desc: "European clinical database", color: "teal", path: "ptbxl", files: ["00001_lr.hea", "00001_lr.dat"] },
  { dataset: "CODE-15%", label: "CODE-15%", desc: "Brazilian 12-lead database", color: "purple", path: "code15s", files: ["13.hea", "13.dat"] },
];

/* ───────── Feature highlights ───────── */
const FEATURES = [
  { icon: "🧠", title: "Deep Learning", desc: "Vision Transformer architecture" },
  { icon: "📊", title: "Multi-Modal", desc: "2D images + 1D signals" },
  { icon: "🔬", title: "5-Fold Ensemble", desc: "Robust ensemble predictions" },
  { icon: "⚡", title: "Real-Time", desc: "Instant ECG analysis" },
];

/* ───────── Circular gauge component ───────── */
function ProbGauge({ probability, isPositive }) {
  const r = 52;
  const circ = 2 * Math.PI * r;
  const offset = circ - (probability * circ);
  const pct = (probability * 100).toFixed(1);

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-36 h-36">
        <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
          <circle cx="60" cy="60" r={r} fill="none" stroke="#e2e8f0" strokeWidth="8" />
          <circle
            cx="60" cy="60" r={r} fill="none"
            strokeWidth="8" strokeLinecap="round"
            stroke={isPositive ? "url(#gaugeRed)" : "url(#gaugeGreen)"}
            strokeDasharray={circ}
            strokeDashoffset={offset}
            className="animate-gauge"
          />
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
          <span className="text-4xl font-extrabold tracking-tight text-slate-800">{pct}</span>
          <span className="text-xs text-slate-400 font-semibold">%</span>
        </div>
      </div>
    </div>
  );
}

/* ───────── ECG animated line ───────── */
function EcgLine({ className = "" }) {
  return (
    <svg viewBox="0 0 400 60" className={`h-8 ${className}`} preserveAspectRatio="none">
      <path
        d="M0,30 L60,30 L80,10 L100,50 L120,20 L140,40 L160,25 L180,30 L240,30 L260,5 L280,55 L300,15 L320,35 L340,30 L400,30"
        fill="none" stroke="url(#ecgGrad)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
        className="ecg-line"
      />
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

/* ───────── Decorative blob ───────── */
function Blob({ className }) {
  return (
    <div className={`absolute rounded-full blur-3xl pointer-events-none ${className}`} />
  );
}

/* ───────── Main App ───────── */
export default function App() {
  const [files, setFiles] = useState([]);
  const [modelType, setModelType] = useState("hybrid");
  const [age, setAge] = useState("");
  const [sex, setSex] = useState("unknown");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [apiOk, setApiOk] = useState(null);
  const [activeTab, setActiveTab] = useState("analyze"); // analyze | about
  const dropRef = useRef(null);
  const fileInputRef = useRef(null);

  const selectedModel = MODEL_OPTIONS.find((m) => m.id === modelType);

  /* health check */
  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((r) => r.ok && setApiOk(true))
      .catch(() => setApiOk(false));
  }, []);

  /* helpers */
  const hasWFDBPair = () => {
    const names = files.map((f) => f.name.toLowerCase());
    return names.some((n) => n.endsWith(".hea")) && (names.some((n) => n.endsWith(".dat")) || names.some((n) => n.endsWith(".mat")));
  };

  const handleFiles = (newFiles) => {
    setResult(null);
    setError("");
    setFiles((prev) => {
      const existing = new Set(prev.map((f) => f.name));
      return [...prev, ...newFiles.filter((f) => !existing.has(f.name))];
    });
  };

  const removeFile = (name) => setFiles((prev) => prev.filter((f) => f.name !== name));

  const [sampleLoading, setSampleLoading] = useState(null);
  const loadSample = async (sample) => {
    setSampleLoading(sample.dataset);
    setResult(null); setError(""); setFiles([]);
    try {
      const loaded = await Promise.all(
        sample.files.map(async (name) => {
          const res = await fetch(`/samples/${sample.path}/${name}`);
          const blob = await res.blob();
          return new File([blob], name);
        })
      );
      handleFiles(loaded);
    } catch {
      setError("Failed to load sample ECG.");
    } finally {
      setSampleLoading(null);
    }
  };

  /* drag & drop */
  const onDrag = (e) => { e.preventDefault(); e.stopPropagation(); };
  const onDragIn = (e) => { onDrag(e); setDragActive(true); };
  const onDragOut = (e) => { onDrag(e); setDragActive(false); };
  const onDrop = (e) => {
    onDrag(e);
    setDragActive(false);
    if (e.dataTransfer.files?.length) handleFiles(Array.from(e.dataTransfer.files));
  };
  const onDropZoneKey = (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInputRef.current?.click(); }
  };

  /* predict */
  const runPrediction = async () => {
    if (!hasWFDBPair()) {
      setError("Upload a matching .hea + .dat (or .mat) pair.");
      return;
    }
    setLoading(true); setError(""); setResult(null);
    try {
      const fd = new FormData();
      files.forEach((f) => fd.append("files", f));
      fd.append("model_type", modelType);
      if (selectedModel.needsDemographics) {
        fd.append("age", age || "50");
        fd.append("sex", sex);
      }
      const res = await fetch(`${API_BASE}/api/predict`, { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const isPositive = result?.prediction === 1;
  const pct = result ? (result.probability * 100).toFixed(1) : null;

  return (
    <div className="relative min-h-screen">
      {/* ── Decorative background blobs ── */}
      <Blob className="w-[500px] h-[500px] bg-brand-200/40 -top-48 -left-48 animate-float" />
      <Blob className="w-[400px] h-[400px] bg-pastel-mint/50 bottom-20 -right-48 animate-float-slow" />
      <Blob className="w-[300px] h-[300px] bg-pastel-lilac/40 top-1/2 left-1/3 animate-float" />

      {/* ── Main container ── */}
      <div className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 py-8">

        {/* ══════════ NAVIGATION BAR ══════════ */}
        <nav className="card shadow-card px-6 py-4 flex items-center justify-between mb-8 animate-fade-in-up">
          <div className="flex items-center gap-3">
            {/* Logo heart */}
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-brand-400 to-brand-600 shadow-brand">
              <svg viewBox="0 0 24 24" fill="none" className="w-5 h-5 text-white">
                <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" fill="currentColor" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-extrabold tracking-tight">
                <span className="bg-gradient-to-r from-brand-500 to-brand-700 bg-clip-text text-transparent">
                  ChagaSight
                </span>
              </h1>
              <p className="text-[10px] text-slate-400 font-medium -mt-0.5">AI-Powered ECG Analysis</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Tab buttons */}
            <div className="flex items-center bg-surface-100 rounded-lg p-1" role="tablist" aria-label="Page sections">
              {[
                { key: "analyze", label: "Analyze" },
                { key: "about", label: "About" },
              ].map((tab) => (
                <button
                  key={tab.key}
                  role="tab"
                  aria-selected={activeTab === tab.key}
                  onClick={() => setActiveTab(tab.key)}
                  className={`px-3 sm:px-4 py-1.5 text-xs font-semibold rounded-md transition-all duration-200
                    ${activeTab === tab.key
                      ? "bg-white text-brand-600 shadow-sm"
                      : "text-slate-400 hover:text-slate-600"
                    }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* API status */}
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-surface-50 border border-surface-200">
              <span className={`w-2 h-2 rounded-full ${
                apiOk === true ? "bg-medical-green animate-pulse-soft" :
                apiOk === false ? "bg-medical-red" : "bg-medical-orange animate-pulse-soft"
              }`} />
              <span className="text-[10px] text-slate-500 font-semibold">
                {apiOk === true ? "API Live" : apiOk === false ? "Offline" : "Checking…"}
              </span>
            </div>
          </div>
        </nav>

        {/* ══════════ ABOUT TAB ══════════ */}
        {activeTab === "about" && (
          <div className="space-y-6 animate-fade-in-up">
            {/* Hero */}
            <div className="card shadow-card overflow-hidden">
              <div className="flex flex-col md:flex-row items-center gap-6 p-8">
                <div className="flex-1">
                  <h2 className="text-3xl font-extrabold text-slate-800 mb-3">
                    Screening Chagas Disease
                    <br />
                    <span className="bg-gradient-to-r from-brand-500 to-medical-teal bg-clip-text text-transparent">
                      from the ECG
                    </span>
                  </h2>
                  <p className="text-slate-500 leading-relaxed mb-4">
                    ChagaSight uses a dual-pathway Vision Transformer combining 2D ECG contour images
                    and 1D raw signals with patient demographics to detect Chagas cardiomyopathy
                    patterns that are invisible to the human eye.
                  </p>
                  <div className="flex items-center gap-2 text-xs text-slate-400">
                    <span className="px-2 py-1 bg-pastel-blue rounded-md font-semibold text-brand-600">PhysioNet 2025</span>
                    <span className="px-2 py-1 bg-pastel-mint rounded-md font-semibold text-medical-teal">173M Parameters</span>
                    <span className="px-2 py-1 bg-pastel-lilac rounded-md font-semibold text-purple-600">5-Fold CV</span>
                  </div>
                </div>
                <div className="w-full md:w-72 flex-shrink-0">
                  <img src="/medical_hero.png" alt="Medical AI heart illustration" className="w-full rounded-xl shadow-card" />
                </div>
              </div>
            </div>

            {/* Features grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {FEATURES.map((f, i) => (
                <div key={i} className="card shadow-card p-5 text-center hover:shadow-card-hover transition-shadow duration-300 group cursor-default"
                     style={{ animationDelay: `${i * 0.1}s` }}>
                  <div className="text-3xl mb-2 group-hover:scale-110 transition-transform duration-300">{f.icon}</div>
                  <h4 className="font-bold text-sm text-slate-700">{f.title}</h4>
                  <p className="text-[11px] text-slate-400 mt-1">{f.desc}</p>
                </div>
              ))}
            </div>

            {/* Architecture image */}
            <div className="card shadow-card p-6">
              <h3 className="font-bold text-slate-700 mb-3 flex items-center gap-2">
                <span className="w-8 h-8 rounded-lg bg-pastel-blue flex items-center justify-center text-brand-500">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4">
                    <circle cx="12" cy="12" r="10" />
                    <path d="M12 8v4l2 2" />
                  </svg>
                </span>
                How It Works
              </h3>
              <div className="flex flex-col md:flex-row items-center gap-6">
                <div className="flex-1 space-y-3">
                  {[
                    { step: "1", text: "Upload 12-lead ECG recording (.hea + .dat)" },
                    { step: "2", text: "AI processes signals through dual ViT pathways" },
                    { step: "3", text: "5 ensemble models vote on Chagas probability" },
                    { step: "4", text: "View result with clinical risk interpretation" },
                  ].map((s) => (
                    <div key={s.step} className="flex items-center gap-3">
                      <span className="w-7 h-7 rounded-full bg-gradient-to-br from-brand-400 to-brand-600 text-white text-xs font-bold flex items-center justify-center flex-shrink-0 shadow-brand">
                        {s.step}
                      </span>
                      <span className="text-sm text-slate-600">{s.text}</span>
                    </div>
                  ))}
                </div>
                <div className="w-full md:w-64 flex-shrink-0">
                  <img src="/ecg_analysis.png" alt="ECG analysis on screen" className="w-full rounded-xl shadow-card" />
                </div>
              </div>
            </div>

            {/* CTA */}
            <button
              onClick={() => setActiveTab("analyze")}
              className="w-full card shadow-card p-4 text-center hover:shadow-card-hover transition-all duration-300 group"
            >
              <span className="text-sm font-bold text-brand-500 group-hover:text-brand-600 transition-colors">
                ← Start Analyzing ECGs
              </span>
            </button>
          </div>
        )}

        {/* ══════════ ANALYZE TAB ══════════ */}
        {activeTab === "analyze" && (
          <div className="space-y-6 animate-fade-in-up">

            {/* ── Hero banner ── */}
            <div className="card shadow-card overflow-hidden">
              <div className="flex items-center gap-6 p-6 pb-5">
                <div className="flex-1">
                  <h2 className="text-2xl font-extrabold text-slate-800 mb-1">
                    ECG Analysis
                  </h2>
                  <p className="text-sm text-slate-400">Upload a 12-lead ECG to screen for Chagas disease</p>
                </div>
                <EcgLine className="w-48 hidden md:block opacity-60" />
              </div>
            </div>

            {/* ── Two-column layout ── */}
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">

              {/* LEFT COLUMN: Model + Demographics */}
              <div className="lg:col-span-2 space-y-5" role="region" aria-label="Model configuration">

                {/* MODEL SELECTOR */}
                <div className="card shadow-card p-5">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4 text-brand-500">
                      <path d="M4 6h16M4 12h16M4 18h16" />
                    </svg>
                    Select Model
                  </h3>
                  <div className="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-1 gap-3" role="radiogroup" aria-label="Model selection">
                    {MODEL_OPTIONS.map((opt) => {
                      const active = modelType === opt.id;
                      return (
                        <button
                          key={opt.id}
                          role="radio"
                          aria-checked={active}
                          aria-label={`${opt.label}${opt.badge ? ", recommended" : ""}`}
                          onClick={() => { setModelType(opt.id); setResult(null); setError(""); }}
                          className={`relative w-full text-left rounded-xl p-3 sm:p-3 lg:p-4 border-2 transition-all duration-300 group
                            ${active
                              ? "model-active bg-pastel-blue/40"
                              : "border-surface-200 hover:border-brand-200 hover:bg-surface-50"
                            }`}
                        >
                          {opt.badge && (
                            <span className="absolute -top-2.5 right-3 rounded-full bg-gradient-to-r from-brand-500 to-brand-600 px-2.5 py-0.5 text-[9px] font-bold text-white shadow-brand" aria-hidden="true">
                              {opt.badge}
                            </span>
                          )}
                          <div className="flex sm:flex-col lg:flex-row items-start sm:items-center lg:items-start gap-3 sm:gap-2 lg:gap-3">
                            <div className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 transition-colors
                              ${active ? opt.iconBg : "bg-surface-100 group-hover:bg-surface-200"}`} aria-hidden="true">
                              <div className={active ? opt.iconColor : "text-slate-400 group-hover:text-slate-500"}>
                                {opt.icon}
                              </div>
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="font-bold text-sm text-slate-700">{opt.label}</div>
                              <div className="text-[11px] text-slate-400 mt-0.5 leading-tight block sm:hidden lg:block">{opt.description}</div>

                              {/* Metric bars — hidden on tablet to save horizontal space */}
                              <div className="mt-3 space-y-2 block sm:hidden lg:block">
                                <div>
                                  <div className="flex items-center justify-between text-[10px] mb-0.5">
                                    <span className="text-slate-400 font-medium">AUROC</span>
                                    <span className="font-bold text-slate-600">{opt.auroc.toFixed(3)}</span>
                                  </div>
                                  <div className="h-1.5 rounded-full bg-surface-200 overflow-hidden">
                                    <div
                                      className="h-full rounded-full bg-gradient-to-r from-brand-400 to-brand-500 transition-all duration-700"
                                      style={{ width: `${opt.auroc * 100}%` }}
                                    />
                                  </div>
                                </div>
                                <div>
                                  <div className="flex items-center justify-between text-[10px] mb-0.5">
                                    <span className="text-slate-400 font-medium">TPR@5%</span>
                                    <span className="font-bold text-slate-600">{opt.tpr.toFixed(3)}</span>
                                  </div>
                                  <div className="h-1.5 rounded-full bg-surface-200 overflow-hidden">
                                    <div
                                      className="h-full rounded-full bg-gradient-to-r from-medical-teal to-medical-green transition-all duration-700"
                                      style={{ width: `${opt.tpr * 100}%` }}
                                    />
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* DEMOGRAPHICS */}
                {selectedModel.needsDemographics && (
                  <div className="card shadow-card p-5 animate-scale-in">
                    <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4 text-medical-teal">
                        <circle cx="12" cy="7" r="4" />
                        <path d="M5.5 21a6.5 6.5 0 0113 0" />
                      </svg>
                      Patient Demographics
                    </h3>
                    <div className="space-y-4">
                      <div>
                        <label htmlFor="age-input" className="block text-xs font-semibold text-slate-500 mb-1.5">
                          Age <span className="text-slate-300 font-normal">(years)</span>
                        </label>
                        <input
                          id="age-input"
                          type="number" min="0" max="120" placeholder="e.g. 45"
                          value={age} onChange={(e) => setAge(e.target.value)}
                          className="w-full rounded-xl bg-surface-50 border border-surface-200 px-4 py-3 text-sm text-slate-700 placeholder:text-slate-300 transition-all"
                        />
                      </div>
                      <div>
                        <label htmlFor="sex-select" className="block text-xs font-semibold text-slate-500 mb-1.5">
                          Biological Sex
                        </label>
                        <select
                          id="sex-select"
                          value={sex} onChange={(e) => setSex(e.target.value)}
                          className="w-full rounded-xl bg-surface-50 border border-surface-200 px-4 py-3 text-sm text-slate-700 transition-all appearance-none cursor-pointer"
                        >
                          <option value="unknown">Unknown</option>
                          <option value="male">Male</option>
                          <option value="female">Female</option>
                        </select>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* RIGHT COLUMN: Upload + Predict + Result */}
              <div className="lg:col-span-3 space-y-5" role="region" aria-label="ECG upload and results">

                {/* FILE UPLOAD */}
                <div className="card shadow-card p-5">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4 text-brand-500">
                      <path d="M12 16V4m0 0l-4 4m4-4l4 4" />
                      <path d="M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
                    </svg>
                    Upload ECG Recording
                  </h3>

                  {/* Sample ECG picker */}
                  <div className="mb-4">
                    <p className="text-xs text-slate-400 font-semibold uppercase tracking-wider mb-2" id="sample-label">Try a Sample ECG</p>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2" role="group" aria-labelledby="sample-label">
                      {SAMPLE_ECGS.map((s) => (
                        <button
                          key={s.dataset}
                          onClick={() => loadSample(s)}
                          disabled={sampleLoading !== null}
                          aria-label={`Load sample ECG from ${s.label} dataset`}
                          aria-busy={sampleLoading === s.dataset}
                          className={`rounded-xl border-2 p-3 text-left transition-all duration-200 hover:shadow-sm min-h-[56px]
                            ${s.color === "brand" ? "border-brand-200 bg-pastel-blue/40 hover:border-brand-400" :
                              s.color === "teal" ? "border-teal-200 bg-pastel-mint/40 hover:border-teal-400" :
                              "border-purple-200 bg-pastel-lilac/40 hover:border-purple-400"}
                            ${sampleLoading === s.dataset ? "opacity-60 cursor-wait" : "cursor-pointer"}`}
                        >
                          <div className={`text-xs font-extrabold mb-0.5
                            ${s.color === "brand" ? "text-brand-600" : s.color === "teal" ? "text-medical-teal" : "text-purple-600"}`}>
                            {sampleLoading === s.dataset ? "Loading…" : s.label}
                          </div>
                          <div className="text-[11px] text-slate-400 leading-tight">{s.desc}</div>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div
                    ref={dropRef}
                    role="button"
                    tabIndex={0}
                    aria-label="Upload ECG files. Drag and drop or press Enter to browse"
                    onDragEnter={onDragIn} onDragLeave={onDragOut} onDragOver={onDrag} onDrop={onDrop}
                    onClick={() => fileInputRef.current?.click()}
                    onKeyDown={onDropZoneKey}
                    className={`rounded-xl border-2 border-dashed cursor-pointer transition-all duration-300 p-6 sm:p-8 text-center group
                      ${dragActive ? "drop-active" : "border-surface-300 hover:border-brand-300 hover:bg-pastel-blue/30"}`}
                  >
                    <input
                      ref={fileInputRef} type="file" multiple accept=".hea,.dat,.mat"
                      aria-label="Select ECG files"
                      onChange={(e) => handleFiles(Array.from(e.target.files))}
                      className="hidden"
                    />
                    <div className="w-16 h-16 mx-auto rounded-2xl bg-pastel-blue flex items-center justify-center mb-4 group-hover:scale-105 transition-transform">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-8 h-8 text-brand-400">
                        <path d="M12 16V4m0 0l-4 4m4-4l4 4" strokeLinecap="round" strokeLinejoin="round" />
                        <path d="M2 17l.621 2.485A2 2 0 004.561 21h14.878a2 2 0 001.94-1.515L22 17" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </div>
                    <p className="text-sm text-slate-600 font-semibold">
                      Drag & drop WFDB files here
                    </p>
                    <p className="text-xs text-slate-400 mt-1">
                      or <span className="text-brand-500 font-semibold cursor-pointer hover:underline">browse files</span>
                    </p>
                    <p className="text-[11px] text-slate-300 mt-2">.hea + .dat or .mat pair required</p>
                  </div>

                  {/* File list */}
                  {files.length > 0 && (
                    <div className="mt-4 space-y-2">
                      {files.map((f) => (
                        <div key={f.name} className="flex items-center justify-between bg-surface-50 rounded-lg px-4 py-2.5 border border-surface-200">
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-lg bg-pastel-blue flex items-center justify-center">
                              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4 text-brand-500">
                                <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z" />
                                <path d="M14 2v6h6" />
                              </svg>
                            </div>
                            <div>
                              <span className="text-sm text-slate-700 font-medium">{f.name}</span>
                              <span className="text-xs text-slate-400 ml-2">({(f.size / 1024).toFixed(0)} KB)</span>
                            </div>
                          </div>
                          <button
                            onClick={(e) => { e.stopPropagation(); removeFile(f.name); }}
                            aria-label={`Remove ${f.name}`}
                            className="w-10 h-10 rounded-lg hover:bg-red-50 flex items-center justify-center text-slate-300 hover:text-red-400 transition-all">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-4 h-4" aria-hidden="true">
                              <path d="M18 6L6 18M6 6l12 12" />
                            </svg>
                          </button>
                        </div>
                      ))}

                      {/* Validation status */}
                      <div className="flex items-center gap-2 mt-2 px-1">
                        {hasWFDBPair() ? (
                          <>
                            <div className="w-5 h-5 rounded-full bg-medical-green/10 flex items-center justify-center">
                              <svg viewBox="0 0 24 24" fill="currentColor" className="w-3.5 h-3.5 text-medical-green">
                                <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z" />
                              </svg>
                            </div>
                            <span className="text-xs text-medical-green font-semibold">Valid WFDB file pair detected</span>
                          </>
                        ) : (
                          <>
                            <div className="w-5 h-5 rounded-full bg-medical-orange/10 flex items-center justify-center">
                              <svg viewBox="0 0 24 24" fill="currentColor" className="w-3.5 h-3.5 text-medical-orange">
                                <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z" />
                              </svg>
                            </div>
                            <span className="text-xs text-medical-orange font-semibold">Need matching .hea + .dat/.mat pair</span>
                          </>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* PREDICT BUTTON */}
                <button
                  onClick={runPrediction}
                  disabled={loading || files.length === 0}
                  className="w-full relative overflow-hidden rounded-xl py-4 font-bold text-white text-sm tracking-wide transition-all duration-300
                    bg-gradient-to-r from-brand-500 via-brand-600 to-brand-700
                    hover:shadow-brand-lg hover:scale-[1.01]
                    disabled:opacity-40 disabled:hover:shadow-none disabled:hover:scale-100 disabled:cursor-not-allowed
                    active:scale-[0.99]"
                >
                  {!loading && (
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/15 to-transparent animate-shimmer bg-[length:200%_100%]" />
                  )}
                  <span className="relative flex items-center justify-center gap-2.5">
                    {loading ? (
                      <>
                        <svg className="w-5 h-5 animate-spin" viewBox="0 0 24 24" fill="none">
                          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="31.4 31.4" strokeLinecap="round" />
                        </svg>
                        Running Inference…
                      </>
                    ) : (
                      <>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-5 h-5">
                          <path d="M2 12h3l2-6 3 12 2-8 2 4 2-2h6" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                        Analyze ECG for Chagas Risk
                      </>
                    )}
                  </span>
                </button>

                {/* ERROR */}
                {error && (
                  <div className="rounded-xl bg-red-50 border border-red-200 p-4 flex items-start gap-3 animate-scale-in">
                    <div className="w-8 h-8 rounded-lg bg-red-100 flex items-center justify-center flex-shrink-0">
                      <svg viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4 text-medical-red">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
                      </svg>
                    </div>
                    <p className="text-sm text-red-600 font-medium pt-1">{error}</p>
                  </div>
                )}

                {/* RESULT */}
                {result && (
                  <div
                    aria-live="polite"
                    aria-atomic="true"
                    className={`card shadow-elevated p-6 space-y-5 animate-slide-up border-2
                    ${isPositive ? "border-red-200 bg-red-50/50" : "border-emerald-200 bg-emerald-50/50"}`}>

                    {/* Header */}
                    <div className="flex items-center justify-between">
                      <h2 className="text-lg font-extrabold text-slate-800">Prediction Result</h2>
                      <span className={`rounded-full px-4 py-1.5 text-xs font-bold tracking-wide shadow-sm
                        ${isPositive
                          ? "bg-gradient-to-r from-red-500 to-orange-500 text-white shadow-glow-red"
                          : "bg-gradient-to-r from-emerald-500 to-teal-400 text-white shadow-glow-green"
                        }`}>
                        {isPositive ? "⚠ POSITIVE" : "✓ NEGATIVE"}
                      </span>
                    </div>

                    {/* Gauge */}
                    <div className="flex flex-col items-center py-3">
                      <ProbGauge probability={result.probability} isPositive={isPositive} />
                      <p className={`mt-3 text-sm font-bold ${isPositive ? "text-red-500" : "text-emerald-500"}`}>
                        {result.interpretation}
                      </p>
                    </div>

                    {/* Probability bar */}
                    <div className="bg-white rounded-xl p-4 border border-surface-200">
                      <div className="flex justify-between text-xs mb-2">
                        <span className="text-slate-500 font-semibold">Chagas Probability</span>
                        <span className="font-extrabold text-slate-700">{pct}%</span>
                      </div>
                      <div className="h-3 rounded-full bg-surface-200 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-1000
                            ${isPositive
                              ? "bg-gradient-to-r from-red-400 to-orange-400"
                              : "bg-gradient-to-r from-emerald-400 to-teal-400"
                            }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>

                    {/* Metadata */}
                    <div className="grid grid-cols-3 gap-2 sm:gap-3">
                      {[
                        { label: "Record", value: result.record },
                        { label: "Model", value: <span className="capitalize">{result.model_type}{result.folds_used > 1 && <span className="text-slate-300"> ({result.folds_used}F)</span>}</span> },
                        { label: "Threshold", value: result.threshold },
                      ].map((item, i) => (
                        <div key={i} className="bg-white rounded-lg border border-surface-200 p-3 text-center">
                          <div className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">{item.label}</div>
                          <div className="text-xs text-slate-700 font-semibold mt-1">{item.value}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ══════════ FOOTER ══════════ */}
        <footer className="text-center text-[11px] text-slate-400 pt-8 pb-6 space-y-1">
          <p className="font-semibold">ChagaSight v1.0 · Research Prototype</p>
          <p>Not validated for clinical use. Consult a medical professional for diagnosis.</p>
        </footer>
      </div>
    </div>
  );
}
