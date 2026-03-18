import { useState } from "react";

const MODEL_OPTIONS = [
  {
    id: "hybrid",
    label: "Hybrid Ensemble",
    badge: "Best",
    auroc: "0.896",
    tpr: "0.504",
    description: "2D contour + 1D signal + demographics",
    needsDemographics: true,
  },
  {
    id: "2d",
    label: "2D Visual",
    badge: null,
    auroc: "0.844",
    tpr: "0.463",
    description: "ECG contour image only",
    needsDemographics: false,
  },
  {
    id: "1d",
    label: "1D Signal",
    badge: null,
    auroc: "0.828",
    tpr: "0.429",
    description: "Raw ECG signal + demographics",
    needsDemographics: true,
  },
];

export default function App() {
  const [files, setFiles] = useState([]);
  const [modelType, setModelType] = useState("hybrid");
  const [age, setAge] = useState("");
  const [sex, setSex] = useState("unknown");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const selectedModel = MODEL_OPTIONS.find((m) => m.id === modelType);

  const handleFiles = (e) => {
    setResult(null);
    setError("");
    setFiles(Array.from(e.target.files));
  };

  const hasWFDBPair = () => {
    const names = files.map((f) => f.name.toLowerCase());
    const hasHea = names.some((n) => n.endsWith(".hea"));
    const hasDat = names.some((n) => n.endsWith(".dat"));
    const hasMat = names.some((n) => n.endsWith(".mat"));
    return hasHea && (hasDat || hasMat);
  };

  const runPrediction = async () => {
    if (!hasWFDBPair()) {
      setError("Please upload a matching .hea + .dat or .mat WFDB file pair.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      files.forEach((f) => formData.append("files", f));
      formData.append("model_type", modelType);

      if (selectedModel.needsDemographics) {
        formData.append("age", age || "50");
        formData.append("sex", sex);
      }

      const res = await fetch("http://127.0.0.1:5050/api/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const isPositive = result && result.prediction === 1;
  const pct = result ? (result.probability * 100).toFixed(1) : null;

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center px-4 py-10">
      <div className="w-full max-w-xl rounded-2xl bg-white shadow-lg p-8 space-y-6">

        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-slate-800">ChagaSight</h1>
          <p className="mt-1 text-sm text-slate-500">
            Vision Transformer–based ECG screening for Chagas disease (research prototype)
          </p>
        </div>

        {/* Model selector */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Model
          </label>
          <div className="grid grid-cols-3 gap-2">
            {MODEL_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                onClick={() => { setModelType(opt.id); setResult(null); setError(""); }}
                className={`relative rounded-xl border p-3 text-left text-xs transition-all
                  ${modelType === opt.id
                    ? "border-sky-400 bg-sky-50 ring-2 ring-sky-300"
                    : "border-slate-200 hover:border-sky-300"
                  }`}
              >
                {opt.badge && (
                  <span className="absolute top-1.5 right-1.5 rounded-full bg-sky-500 px-1.5 py-0.5 text-[10px] font-semibold text-white">
                    {opt.badge}
                  </span>
                )}
                <div className="font-semibold text-slate-800">{opt.label}</div>
                <div className="mt-0.5 text-slate-500">{opt.description}</div>
                <div className="mt-1 text-slate-400">
                  AUROC {opt.auroc} · TPR@5% {opt.tpr}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Demographics (shown when model needs them) */}
        {selectedModel.needsDemographics && (
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Age <span className="text-slate-400 font-normal">(years, optional)</span>
              </label>
              <input
                type="number"
                min="0"
                max="120"
                placeholder="e.g. 45"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm
                           focus:outline-none focus:ring-2 focus:ring-sky-300"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Sex <span className="text-slate-400 font-normal">(optional)</span>
              </label>
              <select
                value={sex}
                onChange={(e) => setSex(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm
                           bg-white focus:outline-none focus:ring-2 focus:ring-sky-300"
              >
                <option value="unknown">Unknown</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </div>
          </div>
        )}

        {/* File upload */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">
            WFDB ECG files
          </label>
          <input
            type="file"
            multiple
            accept=".hea,.dat,.mat"
            onChange={handleFiles}
            className="block w-full text-sm text-slate-600
              file:mr-4 file:py-2 file:px-4
              file:rounded-xl file:border-0
              file:text-sm file:font-semibold
              file:bg-sky-50 file:text-sky-600
              hover:file:bg-sky-100"
          />
          {files.length > 0 && (
            <div className="mt-2 space-y-0.5 text-xs text-slate-500">
              {files.map((f) => (
                <div key={f.name}>· {f.name}</div>
              ))}
            </div>
          )}
        </div>

        {/* Predict button */}
        <button
          onClick={runPrediction}
          disabled={loading}
          className="w-full rounded-xl bg-gradient-to-r from-sky-500 to-indigo-500
                     py-3 font-semibold text-white
                     hover:opacity-90 disabled:opacity-50 transition-opacity"
        >
          {loading ? "Running inference…" : "Predict Chagas Risk"}
        </button>

        {/* Error */}
        {error && (
          <div className="rounded-xl bg-red-50 border border-red-100 p-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* Result */}
        {result && (
          <div className={`rounded-xl border p-5 space-y-3
            ${isPositive ? "border-rose-200 bg-rose-50" : "border-emerald-200 bg-emerald-50"}`}
          >
            <div className="flex items-center justify-between">
              <h2 className="font-semibold text-slate-800">Prediction Result</h2>
              <span className={`rounded-full px-3 py-1 text-sm font-bold
                ${isPositive
                  ? "bg-rose-500 text-white"
                  : "bg-emerald-500 text-white"}`}
              >
                {isPositive ? "Positive" : "Negative"}
              </span>
            </div>

            {/* Probability bar */}
            <div>
              <div className="flex justify-between text-xs text-slate-500 mb-1">
                <span>Chagas probability</span>
                <span className="font-semibold text-slate-700">{pct}%</span>
              </div>
              <div className="h-3 w-full rounded-full bg-slate-200 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${isPositive ? "bg-rose-500" : "bg-emerald-500"}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>

            <p className={`text-sm font-medium ${isPositive ? "text-rose-700" : "text-emerald-700"}`}>
              {result.interpretation}
            </p>

            {/* Metadata */}
            <div className="text-xs text-slate-400 space-y-0.5 border-t border-slate-200 pt-3 mt-1">
              <div>Record: <span className="text-slate-600">{result.record}</span></div>
              <div>Model: <span className="text-slate-600">{result.model_type}</span>
                {result.folds_used > 1 && (
                  <span className="text-slate-400"> ({result.folds_used} folds)</span>
                )}
              </div>
              <div>Threshold: <span className="text-slate-600">{result.threshold}</span></div>
            </div>
          </div>
        )}

        {/* Disclaimer */}
        <p className="text-xs text-slate-400">
          Research prototype only. Not validated for clinical use.
        </p>
      </div>
    </div>
  );
}
