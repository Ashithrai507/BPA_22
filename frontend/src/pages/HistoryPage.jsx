import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api';

export default function HistoryPage() {
  const [history, setHistory] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const limit = 20;

  const fetchHistory = async (offset = 0) => {
    setLoading(true);
    try {
      const resp = await api.get(`/predict/history?limit=${limit}&offset=${offset}`);
      setHistory(resp.data.items);
      setTotal(resp.data.total);
    } catch {}
    setLoading(false);
  };

  useEffect(() => { fetchHistory(page * limit); }, [page]);

  const handleDelete = async (id) => {
    if (!confirm('Delete this prediction?')) return;
    await api.delete(`/predict/${id}`);
    fetchHistory(page * limit);
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'complete': return '\u2705';
      case 'pending': return '\u23F3';
      case 'failed': return '\u274C';
      default: return '\u2796';
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Pipeline History</h2>

      {loading && <p className="text-blue-600">Loading...</p>}

      <div className="bg-white rounded-lg shadow divide-y">
        {history.map((h) => (
          <div key={h.id} className="px-4 py-3 flex items-center justify-between hover:bg-gray-50">
            <button
              onClick={() => navigate(`/predict/${h.id}`)}
              className="flex-1 text-left"
            >
              <div className="flex items-center gap-4">
                <span className="text-sm font-medium">{h.predicted_species || 'unknown'}</span>
                <span className="text-xs text-gray-500">{h.filename}</span>
                <span className="text-xs text-gray-400">#{h.id}</span>
                <span className="text-xs" title={`Protein analysis: ${h.protein_status}`}>
                  {getStatusIcon(h.protein_status)}
                </span>
              </div>
              <div className="text-xs text-gray-400 mt-1">
                {h.confidence?.toFixed(2) ?? '\u2014'} confidence — {h.created_at}
              </div>
            </button>
            <button
              onClick={() => handleDelete(h.id)}
              className="text-red-500 hover:text-red-700 text-sm px-2"
            >
              Delete
            </button>
          </div>
        ))}
      </div>

      <div className="flex justify-between items-center">
        <button
          onClick={() => setPage(p => Math.max(0, p - 1))}
          disabled={page === 0}
          className="px-4 py-2 bg-gray-200 rounded disabled:opacity-50"
        >
          Previous
        </button>
        <span className="text-sm text-gray-600">
          Page {page + 1} of {Math.ceil(total / limit)}
        </span>
        <button
          onClick={() => setPage(p => p + 1)}
          disabled={(page + 1) * limit >= total}
          className="px-4 py-2 bg-gray-200 rounded disabled:opacity-50"
        >
          Next
        </button>
      </div>
    </div>
  );
}
