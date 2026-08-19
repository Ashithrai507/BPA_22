import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import SpeciesSelector from '../components/SpeciesSelector';
import ProteinTable from '../components/ProteinTable';
import FastaDownload from '../components/FastaDownload';
import api from '../api';

export default function ProteinPage() {
  const location = useLocation();
  const [species, setSpecies] = useState('');
  const [topN, setTopN] = useState(10);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [batchMode, setBatchMode] = useState(false);
  const [batchSpecies, setBatchSpecies] = useState('');
  const [batchResults, setBatchResults] = useState([]);

  const [history, setHistory] = useState([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historyPage, setHistoryPage] = useState(0);
  const [showHistory, setShowHistory] = useState(false);
  const pageSize = 10;

  const handleRun = async () => {
    if (!species) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await api.post('/proteins', { species, top_n: topN });
      setResult(resp.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleBatchRun = async () => {
    const speciesArray = batchSpecies.split('\n').filter(s => s.trim());
    if (speciesArray.length === 0) return;
    setLoading(true);
    setError(null);
    setBatchResults([]);
    try {
      const resp = await api.post(`/proteins/batch?top_n=${topN}`, speciesArray);
      setBatchResults(resp.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadHistory = async (page) => {
    try {
      const resp = await api.get(`/proteins/history?limit=${pageSize}&offset=${page * pageSize}`);
      setHistory(resp.data.items);
      setHistoryTotal(resp.data.total);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
  };

  useEffect(() => {
    if (showHistory) loadHistory(historyPage);
  }, [showHistory, historyPage]);

  useEffect(() => {
    const incoming = location.state?.species;
    if (incoming) {
      setSpecies(incoming);
      setShowHistory(false);
      setBatchMode(false);

      api.post('/proteins', { species: incoming, top_n: topN })
        .then((resp) => setResult(resp.data))
        .catch((err) => setError(err.response?.data?.detail || err.message))
        .finally(() => setLoading(false));
    }
  }, [location.state]);

  const historyPages = Math.ceil(historyTotal / pageSize);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-800">Protein Ranking</h2>
        <div className="flex gap-2">
          <button
            onClick={() => { setBatchMode(false); setShowHistory(false); }}
            className={`px-4 py-2 rounded text-sm font-semibold ${!batchMode && !showHistory ? 'bg-green-600 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}`}
          >
            Single
          </button>
          <button
            onClick={() => { setBatchMode(true); setShowHistory(false); }}
            className={`px-4 py-2 rounded text-sm font-semibold ${batchMode ? 'bg-green-600 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}`}
          >
            Batch
          </button>
          <button
            onClick={() => { setShowHistory(!showHistory); setBatchMode(false); }}
            className={`px-4 py-2 rounded text-sm font-semibold ${showHistory ? 'bg-green-600 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}`}
          >
            History
          </button>
        </div>
      </div>

      {!batchMode && !showHistory && (
        <>
          <div className="flex items-end gap-4">
            <div>
              <label className="block text-sm text-gray-600 mb-1">Species</label>
              <SpeciesSelector value={species} onChange={setSpecies} />
            </div>
            <div>
              <label className="block text-sm text-gray-600 mb-1">Top N</label>
              <input type="number" value={topN} onChange={(e) => setTopN(Number(e.target.value))} min={1} max={50} className="border rounded px-3 py-2 text-sm w-20" />
            </div>
            <button onClick={handleRun} disabled={!species || loading} className="bg-green-600 text-white px-6 py-2 rounded hover:bg-green-700 font-semibold disabled:opacity-50">
              {loading ? 'Running...' : 'Run Ranking'}
            </button>
          </div>

          {error && <p className="text-red-600 text-sm">{error}</p>}

          {result && (
            <>
              <div className="text-sm text-gray-600">
                {result.resolved_name} — taxonomy ID {result.taxonomy_id} — source: {result.source}
              </div>
              <ProteinTable proteins={result.selected_proteins} />
              <FastaDownload proteins={result.selected_proteins} species={result.species} />

              {result.excluded?.length > 0 && (
                <p className="text-sm text-gray-500">{result.excluded.length} proteins excluded (too short or non-standard residues)</p>
              )}
            </>
          )}
        </>
      )}

      {batchMode && (
        <>
          <div className="flex items-end gap-4">
            <div className="flex-1">
              <label className="block text-sm text-gray-600 mb-1">Species (one per line)</label>
              <textarea
                value={batchSpecies}
                onChange={(e) => setBatchSpecies(e.target.value)}
                rows={5}
                className="border rounded px-3 py-2 text-sm w-full font-mono"
                placeholder={"Bacillus subtilis\nEscherichia coli\nStaphylococcus aureus"}
              />
            </div>
            <div>
              <label className="block text-sm text-gray-600 mb-1">Top N</label>
              <input type="number" value={topN} onChange={(e) => setTopN(Number(e.target.value))} min={1} max={50} className="border rounded px-3 py-2 text-sm w-20" />
            </div>
            <button onClick={handleBatchRun} disabled={loading} className="bg-green-600 text-white px-6 py-2 rounded hover:bg-green-700 font-semibold disabled:opacity-50">
              {loading ? 'Running...' : 'Run Batch'}
            </button>
          </div>

          {error && <p className="text-red-600 text-sm">{error}</p>}

          {batchResults.length > 0 && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-700">Batch Results ({batchResults.length} species)</h3>
              {batchResults.map((r, i) => (
                <div key={i} className="border rounded p-4 bg-gray-50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-gray-800">{r.resolved_name}</span>
                    <span className="text-xs text-gray-500">taxonomy {r.taxonomy_id} — {r.source}</span>
                  </div>
                  <ProteinTable proteins={r.selected_proteins} />
                  <FastaDownload proteins={r.selected_proteins} species={r.species} />
                  {r.excluded?.length > 0 && (
                    <p className="text-xs text-gray-500 mt-1">{r.excluded.length} proteins excluded</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {showHistory && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-700">Protein Analysis History</h3>
          {history.length === 0 ? (
            <p className="text-gray-500 text-sm">No protein analyses yet.</p>
          ) : (
            <>
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="bg-gray-100 text-left">
                    <th className="p-2 border">ID</th>
                    <th className="p-2 border">Species</th>
                    <th className="p-2 border">Image Species</th>
                    <th className="p-2 border">Status</th>
                    <th className="p-2 border">Created</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((item) => (
                    <tr key={item.id} className="hover:bg-gray-50">
                      <td className="p-2 border">{item.id}</td>
                      <td className="p-2 border">{item.species}</td>
                      <td className="p-2 border">{item.image_species || '—'}</td>
                      <td className="p-2 border">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${item.status === 'complete' ? 'bg-green-100 text-green-800' : item.status === 'error' ? 'bg-red-100 text-red-800' : 'bg-yellow-100 text-yellow-800'}`}>
                          {item.status}
                        </span>
                      </td>
                      <td className="p-2 border">{item.created_at}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {historyPages > 1 && (
                <div className="flex items-center gap-2 text-sm">
                  <button onClick={() => setHistoryPage(p => Math.max(0, p - 1))} disabled={historyPage === 0} className="px-3 py-1 rounded bg-gray-200 hover:bg-gray-300 disabled:opacity-50">Prev</button>
                  <span className="text-gray-600">Page {historyPage + 1} of {historyPages}</span>
                  <button onClick={() => setHistoryPage(p => Math.min(historyPages - 1, p + 1))} disabled={historyPage >= historyPages - 1} className="px-3 py-1 rounded bg-gray-200 hover:bg-gray-300 disabled:opacity-50">Next</button>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
