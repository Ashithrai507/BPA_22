import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ImageUpload from '../components/ImageUpload';
import ResultsPanel from '../components/ResultsPanel';
import ResultsTable from '../components/ResultsTable';
import api from '../api';

export default function PredictPage() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [showJson, setShowJson] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    api.get('/predict/history?limit=10').then((r) => setHistory(r.data.items)).catch(() => {});
  }, []);

  const loadHistory = async (id) => {
    try {
      const resp = await api.get(`/predict/${id}`);
      setResult(resp.data);
    } catch {}
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Image Analysis</h2>

      <ImageUpload onResult={(r) => { setResult(r); api.get('/predict/history?limit=10').then((r) => setHistory(r.data.items)).catch(() => {}); }} onLoading={setLoading} />

      {loading && <p className="text-blue-600">Analyzing...</p>}

      {result && (
        <>
          <ResultsPanel result={result} />
          <ResultsTable colonies={result.colonies} />

          <div className="flex items-center gap-3">
            <button onClick={() => setShowJson(!showJson)} className="text-sm text-blue-600 hover:underline">
              {showJson ? 'Hide' : 'Show'} Raw JSON
            </button>
            <button
              onClick={() => navigate('/proteins', { state: { species: result.predicted_species } })}
              className="bg-green-600 text-white px-5 py-2 rounded hover:bg-green-700 font-semibold text-sm"
            >
              Protein Analysis
            </button>
          </div>

          {showJson && (
            <pre className="mt-2 bg-gray-900 text-green-400 p-4 rounded-lg text-xs overflow-auto max-h-96">
              {JSON.stringify(result, null, 2)}
            </pre>
          )}
        </>
      )}

      {history.length > 0 && (
        <div className="mt-8">
          <h3 className="text-sm font-semibold text-gray-600 mb-2">Recent Predictions</h3>
          <div className="bg-white rounded-lg shadow divide-y">
            {history.map((h) => (
              <button key={h.id} onClick={() => loadHistory(h.id)} className="w-full text-left px-4 py-3 hover:bg-gray-50 flex justify-between">
                <span className="text-sm font-medium">{h.predicted_species || 'unknown'}</span>
                <span className="text-xs text-gray-500">
                  {h.protein_status === 'complete' ? '\u2705' : '\u2796'} #{h.id}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
