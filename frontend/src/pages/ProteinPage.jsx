import { useState } from 'react';
import SpeciesSelector from '../components/SpeciesSelector';
import ProteinTable from '../components/ProteinTable';
import FastaDownload from '../components/FastaDownload';
import api from '../api';

export default function ProteinPage() {
  const [species, setSpecies] = useState('');
  const [topN, setTopN] = useState(10);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Protein Ranking</h2>

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
    </div>
  );
}
