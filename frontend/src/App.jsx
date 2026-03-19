import { useState, useRef, useEffect } from "react";

/* ───────── Model options ───────── */
const MODEL_OPTIONS = [
  {
    id: "hybrid",
    label: "Hybrid Ensemble",
    badge: "Recommended",
    auroc: 0.896,
    tpr: 0.504,
    description: "2D contour + 1D signal + demographics",
    needsDemographics: true,
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
    label: "2D Visual",
    badge: null,
    auroc: 0.844,
    tpr: 0.463,
    description: "ECG contour image analysis",
    needsDemographics: false,
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <rect x="3" y="3" width="18" height="18" rx="3" />
        <path d="M3 15l4-4 3 3 4-5 7 6" />
      </svg>
    ),
  },
  {
    id: "1d",
    label: "1D Signal",
    badge: null,
    auroc: 0.828,
    tpr: 0.429,
    description: "Raw ECG signal + demographics",
    needsDemographics: true,
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <path d="M2 12h3l2-6 3 12 2-8 2 4 2-2h6" />
      </svg>
    ),
  },
];

/* ───────── Circular gauge component ───────── */
function ProbGauge({ probability, isPositive }) {
  const r = 48;
  const circ = 2 * Math.PI * r;
  const offset = circ - (probability * circ);
  const pct = (probability * 100).toFixed(1);

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-32 h-32">
        <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
          <circle cx="60" cy="60" r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="8" />
          <circle
            cx="60" cy="60" r={r} fill="none"
            strokeWidth="8" strokeLinecap="round"
            stroke={isPositive ? "url(#gaugeRed)" : "url(#gaugeGreen)"}
            strokeDasharray={circ}
            strokeDashoffset={offset}
            className="animate-gauge"
            style={{ "--gauge-target": offset }}
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
          <span className="text-3xl font-extrabold tracking-tight">{pct}</span>
          <span className="text-xs text-white/50 font-medium">%</span>
        </div>
      </div>
    </div>
  );
}

/* ───────── ECG header icon ───────── */
function EcgIcon() {
  return (
    <svg viewBox="0 0 200 60" className="w-28 h-8 text-brand-400">
      <path
        d="M0,30 L30,30 L40,10 L50,50 L60,20 L70,40 L80,25 L90,30 L120,30 L130,5 L140,55 L150,15 L160,35 L170,30 L200,30"
        fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"
        className="ecg-line"
      />
    </svg>
  );
}

/* ───────── Floating orb ───────── */
function Orb({ className }) {
  return (
    <div className={`absolute rounded-full blur-3xl opacity-20 pointer-events-none ${className}`} />
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
  const dropRef = useRef(null);
  const fileInputRef = useRef(null);

  const selectedModel = MODEL_OPTIONS.find((m) => m.id === modelType);

  /* health check */
  useEffect(() => {
    fetch("http://127.0.0.1:5050/api/health")
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
      const merged = [...prev, ...newFiles.filter((f) => !existing.has(f.name))];
      return merged;
    });
  };

  const removeFile = (name) => setFiles((prev) => prev.filter((f) => f.name !== name));

  /* drag & drop */
  const onDrag = (e) => { e.preventDefault(); e.stopPropagation(); };
  const onDragIn = (e) => { onDrag(e); setDragActive(true); };
  const onDragOut = (e) => { onDrag(e); setDragActive(false); };
  const onDrop = (e) => {
    onDrag(e);
    setDragActive(false);
    if (e.dataTransfer.files?.length) handleFiles(Array.from(e.dataTransfer.files));
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
      const res = await fetch("http://127.0.0.1:5050/api/predict", { method: "POST", body: fd });
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
    <div className="relative min-h-screen overflow-hidden">
      {/* ── Animated background orbs ── */}
      <Orb className="w-96 h-96 bg-brand-500 top-[-10%] left-[-10%] animate-float" />
      <Orb className="w-80 h-80 bg-purple-600 bottom-[5%] right-[-8%] animate-float-slow" />
      <Orb className="w-64 h-64 bg-teal-500 top-[40%] right-[20%] animate-float" />

      {/* ── Main container ── */}
      <div className="relative z-10 flex items-center justify-center min-h-screen px-4 py-10">
        <div className="w-full max-w-2xl space-y-6 animate-fade-in-up">

          {/* ══════════ HEADER ══════════ */}
          <header className="glass rounded-2xl p-6 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-brand-500/20 ring-1 ring-brand-400/30">
                <svg viewBox="0 0 24 24" fill="none" className="w-7 h-7 text-brand-400">
                  <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" fill="currentColor" />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-extrabold tracking-tight">
                  <span className="bg-gradient-to-r from-brand-400 to-purple-400 bg-clip-text text-transparent">
                    ChagaSight
                  </span>
                </h1>
                <p className="text-xs text-white/40 mt-0.5">AI-Powered Chagas Disease Detection</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <EcgIcon />
              <div className="flex items-center gap-1.5">
                <span className={`w-2 h-2 rounded-full ${apiOk === true ? "bg-emerald-400 animate-pulse-glow" : apiOk === false ? "bg-red-400" : "bg-yellow-400 animate-pulse-glow"}`} />
                <span className="text-[10px] text-white/40 font-medium">
                  {apiOk === true ? "API Live" : apiOk === false ? "Offline" : "Checking…"}
                </span>
              </div>
            </div>
          </header>

          {/* ══════════ MODEL SELECTOR ══════════ */}
          <section>
            <label className="block text-sm font-semibold text-white/60 mb-3 tracking-wide uppercase">
              Select Model
            </label>
            <div className="grid grid-cols-3 gap-3">
              {MODEL_OPTIONS.map((opt) => {
                const active = modelType === opt.id;
                return (
                  <button
                    key={opt.id}
                    onClick={() => { setModelType(opt.id); setResult(null); setError(""); }}
                    className={`relative glass rounded-xl p-4 text-left transition-all duration-300 group
                      ${active
                        ? "ring-2 ring-brand-400/70 bg-brand-500/10 shadow-[0_0_20px_rgba(51,131,252,0.15)]"
                        : "hover:bg-white/[0.04] hover:ring-1 hover:ring-white/10"
                      }`}
                  >
                    {opt.badge && (
                      <span className="absolute -top-2 -right-2 rounded-full bg-gradient-to-r from-brand-500 to-purple-500 px-2.5 py-0.5 text-[9px] font-bold text-white shadow-lg shadow-brand-500/30">
                        {opt.badge}
                      </span>
                    )}
                    <div className={`mb-2 transition-colors ${active ? "text-brand-400" : "text-white/30 group-hover:text-white/50"}`}>
                      {opt.icon}
                    </div>
                    <div className="font-bold text-sm text-white/90">{opt.label}</div>
                    <div className="text-[11px] text-white/35 mt-1 leading-tight">{opt.description}</div>
                    <div className="mt-3 space-y-1.5">
                      <div className="flex items-center justify-between text-[10px] text-white/40">
                        <span>AUROC</span>
                        <span className="font-semibold text-white/60">{opt.auroc.toFixed(3)}</span>
                      </div>
                      <div className="h-1 rounded-full bg-white/[0.06] overflow-hidden">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-brand-400 to-brand-600 transition-all duration-700"
                          style={{ width: `${opt.auroc * 100}%` }}
                        />
                      </div>
                      <div className="flex items-center justify-between text-[10px] text-white/40">
                        <span>TPR@5%</span>
                        <span className="font-semibold text-white/60">{opt.tpr.toFixed(3)}</span>
                      </div>
                      <div className="h-1 rounded-full bg-white/[0.06] overflow-hidden">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-purple-400 to-purple-600 transition-all duration-700"
                          style={{ width: `${opt.tpr * 100}%` }}
                        />
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </section>

          {/* ══════════ DEMOGRAPHICS ══════════ */}
          {selectedModel.needsDemographics && (
            <section className="glass rounded-xl p-5 space-y-4 animate-fade-in-up">
              <h3 className="text-sm font-semibold text-white/60 tracking-wide uppercase flex items-center gap-2">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4 text-brand-400">
                  <circle cx="12" cy="7" r="4" />
                  <path d="M5.5 21a6.5 6.5 0 0113 0" />
                </svg>
                Patient Demographics
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-white/40 mb-1.5">
                    Age <span className="text-white/20">(years)</span>
                  </label>
                  <input
                    type="number" min="0" max="120" placeholder="e.g. 45"
                    value={age} onChange={(e) => setAge(e.target.value)}
                    className="w-full rounded-lg bg-white/[0.04] border border-white/[0.08] px-3.5 py-2.5 text-sm text-white placeholder:text-white/20 focus:outline-none focus:ring-2 focus:ring-brand-400/40 focus:border-brand-400/40 transition-all"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-white/40 mb-1.5">
                    Sex
                  </label>
                  <select
                    value={sex} onChange={(e) => setSex(e.target.value)}
                    className="w-full rounded-lg bg-white/[0.04] border border-white/[0.08] px-3.5 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-brand-400/40 focus:border-brand-400/40 transition-all appearance-none cursor-pointer"
                  >
                    <option value="unknown" className="bg-[#1a1040]">Unknown</option>
                    <option value="male" className="bg-[#1a1040]">Male</option>
                    <option value="female" className="bg-[#1a1040]">Female</option>
                  </select>
                </div>
              </div>
            </section>
          )}

          {/* ══════════ FILE UPLOAD ══════════ */}
          <section>
            <h3 className="text-sm font-semibold text-white/60 mb-3 tracking-wide uppercase flex items-center gap-2">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4 text-brand-400">
                <path d="M12 16V4m0 0l-4 4m4-4l4 4" />
                <path d="M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
              </svg>
              Upload ECG Files
            </h3>
            <div
              ref={dropRef}
              onDragEnter={onDragIn} onDragLeave={onDragOut} onDragOver={onDrag} onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`glass rounded-xl border-2 border-dashed cursor-pointer transition-all duration-300 p-8 text-center
                ${dragActive ? "drop-active" : "border-white/[0.08] hover:border-white/20 hover:bg-white/[0.03]"}`}
            >
              <input
                ref={fileInputRef} type="file" multiple accept=".hea,.dat,.mat"
                onChange={(e) => handleFiles(Array.from(e.target.files))}
                className="hidden"
              />
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-10 h-10 mx-auto text-white/20 mb-3">
                <path d="M12 16V4m0 0l-4 4m4-4l4 4" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M2 17l.621 2.485A2 2 0 004.561 21h14.878a2 2 0 001.94-1.515L22 17" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <p className="text-sm text-white/40 font-medium">
                Drag & drop WFDB files here, or <span className="text-brand-400 underline underline-offset-2">browse</span>
              </p>
              <p className="text-xs text-white/20 mt-1">.hea + .dat or .mat pair required</p>
            </div>

            {/* file list */}
            {files.length > 0 && (
              <div className="mt-3 space-y-1.5">
                {files.map((f) => (
                  <div key={f.name} className="flex items-center justify-between glass rounded-lg px-3 py-2 text-xs">
                    <div className="flex items-center gap-2">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4 text-brand-400/60">
                        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z" />
                        <path d="M14 2v6h6" />
                      </svg>
                      <span className="text-white/60">{f.name}</span>
                      <span className="text-white/20">({(f.size / 1024).toFixed(0)} KB)</span>
                    </div>
                    <button onClick={(e) => { e.stopPropagation(); removeFile(f.name); }}
                      className="text-white/20 hover:text-red-400 transition-colors p-1">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3.5 h-3.5">
                        <path d="M18 6L6 18M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                ))}
                <div className="flex items-center gap-1.5 text-xs mt-1 px-1">
                  {hasWFDBPair() ? (
                    <>
                      <svg viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4 text-emerald-400">
                        <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z" />
                      </svg>
                      <span className="text-emerald-400/80 font-medium">Valid WFDB file pair</span>
                    </>
                  ) : (
                    <>
                      <svg viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4 text-amber-400">
                        <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z" />
                      </svg>
                      <span className="text-amber-400/80 font-medium">Need .hea + .dat/.mat pair</span>
                    </>
                  )}
                </div>
              </div>
            )}
          </section>

          {/* ══════════ PREDICT BUTTON ══════════ */}
          <button
            onClick={runPrediction}
            disabled={loading || files.length === 0}
            className="w-full relative overflow-hidden rounded-xl py-3.5 font-bold text-white text-sm tracking-wide transition-all duration-300
              bg-gradient-to-r from-brand-500 via-brand-600 to-purple-600
              hover:shadow-[0_0_30px_rgba(51,131,252,0.3)] hover:scale-[1.01]
              disabled:opacity-40 disabled:hover:shadow-none disabled:hover:scale-100 disabled:cursor-not-allowed
              active:scale-[0.99]"
          >
            {/* shimmer overlay */}
            {!loading && (
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-shimmer bg-[length:200%_100%]" />
            )}
            <span className="relative flex items-center justify-center gap-2">
              {loading ? (
                <>
                  <svg className="w-5 h-5 animate-spin" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="31.4 31.4" strokeLinecap="round" />
                  </svg>
                  Running inference…
                </>
              ) : (
                <>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-5 h-5">
                    <path d="M2 12h3l2-6 3 12 2-8 2 4 2-2h6" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  Predict Chagas Risk
                </>
              )}
            </span>
          </button>

          {/* ══════════ ERROR ══════════ */}
          {error && (
            <div className="rounded-xl bg-red-500/10 border border-red-500/20 p-4 flex items-start gap-3 animate-fade-in-up">
              <svg viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
              </svg>
              <p className="text-sm text-red-300">{error}</p>
            </div>
          )}

          {/* ══════════ RESULT ══════════ */}
          {result && (
            <section className={`rounded-2xl p-6 space-y-5 animate-fade-in-up border ${isPositive
                ? "bg-red-500/[0.06] border-red-500/20"
                : "bg-emerald-500/[0.06] border-emerald-500/20"
              }`}>
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-bold text-white/90">Prediction Result</h2>
                <span className={`rounded-full px-4 py-1.5 text-xs font-bold tracking-wide ${isPositive
                    ? "bg-gradient-to-r from-red-500 to-orange-500 text-white shadow-lg shadow-red-500/20"
                    : "bg-gradient-to-r from-emerald-500 to-teal-400 text-white shadow-lg shadow-emerald-500/20"
                  }`}>
                  {isPositive ? "⚠ POSITIVE" : "✓ NEGATIVE"}
                </span>
              </div>

              {/* gauge */}
              <div className="flex flex-col items-center py-2">
                <ProbGauge probability={result.probability} isPositive={isPositive} />
                <p className={`mt-3 text-sm font-semibold ${isPositive ? "text-red-400" : "text-emerald-400"}`}>
                  {result.interpretation}
                </p>
              </div>

              {/* probability bar */}
              <div>
                <div className="flex justify-between text-xs text-white/40 mb-1.5">
                  <span>Chagas Probability</span>
                  <span className="font-bold text-white/70">{pct}%</span>
                </div>
                <div className="h-2 rounded-full bg-white/[0.06] overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-1000 ${isPositive
                        ? "bg-gradient-to-r from-red-500 to-orange-400"
                        : "bg-gradient-to-r from-emerald-500 to-teal-400"
                      }`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>

              {/* metadata */}
              <div className="grid grid-cols-3 gap-3 pt-3 border-t border-white/[0.06]">
                <div className="text-center">
                  <div className="text-[10px] text-white/30 uppercase tracking-wider">Record</div>
                  <div className="text-xs text-white/70 font-semibold mt-1">{result.record}</div>
                </div>
                <div className="text-center">
                  <div className="text-[10px] text-white/30 uppercase tracking-wider">Model</div>
                  <div className="text-xs text-white/70 font-semibold mt-1 capitalize">{result.model_type}
                    {result.folds_used > 1 && <span className="text-white/30"> ({result.folds_used}F)</span>}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-[10px] text-white/30 uppercase tracking-wider">Threshold</div>
                  <div className="text-xs text-white/70 font-semibold mt-1">{result.threshold}</div>
                </div>
              </div>
            </section>
          )}

          {/* ══════════ FOOTER ══════════ */}
          <footer className="text-center text-[11px] text-white/20 pt-2 pb-4">
            Research prototype — not validated for clinical use. · ChagaSight v1.0
          </footer>
        </div>
      </div>
    </div>
  );
}
