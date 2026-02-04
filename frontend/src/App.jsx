import { useState } from "react";

export default function App() {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleFiles = (e) => {
    setResult(null);
    setError("");
    setFiles(Array.from(e.target.files));
  };

  const hasWFDBPair = () => {
    const names = files.map(f => f.name.toLowerCase());
    const hasHea = names.some(n => n.endsWith(".hea"));
    const hasDat = names.some(n => n.endsWith(".dat"));
    const hasMat = names.some(n => n.endsWith(".mat"));
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
      files.forEach(f => formData.append("files", f));

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

  return (
    <div className="min-h-screen bg-mint-50 flex items-center justify-center px-4">
      <div className="w-full max-w-xl rounded-2xl bg-white shadow-lg p-8">
        <h1 className="text-3xl font-bold text-lilac-500">
          ChagaSight
        </h1>
        <p className="mt-2 text-slate-600">
          Vision Transformer–based ECG screening (research prototype)
        </p>

        {/* Upload */}
        <div className="mt-6">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Upload WFDB ECG files
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
              file:bg-mint-100 file:text-mint-600
              hover:file:bg-mint-200"
          />
        </div>

        {/* File list */}
        {files.length > 0 && (
          <div className="mt-4 space-y-1 text-sm text-slate-700">
            {files.map((f) => (
              <div key={f.name}>• {f.name}</div>
            ))}
          </div>
        )}

        {/* Action */}
        <button
          onClick={runPrediction}
          disabled={loading}
          className="mt-6 w-full rounded-xl bg-gradient-to-r from-mint-400 to-sky-400
                     py-3 font-semibold text-slate-900
                     hover:opacity-90 disabled:opacity-60"
        >
          {loading ? "Running inference…" : "Predict Chagas Risk"}
        </button>

        {/* Error */}
        {error && (
          <div className="mt-4 rounded-xl bg-red-50 p-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="mt-6 rounded-xl border border-slate-200 p-4">
            <h2 className="font-semibold text-slate-800">
              Prediction Result
            </h2>

            <div className="mt-3 text-2xl font-bold text-slate-900">
              {(result.probability * 100).toFixed(2)}%
            </div>

            <div className="mt-1 text-sm text-slate-600">
              {result.interpretation}
            </div>

            <div className="mt-3 text-xs text-slate-500">
              Threshold: {result.threshold}
            </div>
          </div>
        )}

        {/* Disclaimer */}
        <div className="mt-6 text-xs text-slate-500">
          
        </div>
      </div>
    </div>
  );
}
