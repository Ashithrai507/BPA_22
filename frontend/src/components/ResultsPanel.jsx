export default function ResultsPanel({ result }) {
  if (!result) return null;

  const conf = typeof result.confidence === 'number' ? result.confidence : 0;
  const confColor = conf > 0.75 ? 'text-green-700' : conf > 0.5 ? 'text-orange-600' : 'text-red-600';

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
      <div className="bg-white rounded-lg p-4 shadow">
        <p className="text-xs text-gray-500 uppercase">Species</p>
        <p className="text-lg font-semibold">{result.predicted_species || '—'}</p>
      </div>
      <div className="bg-white rounded-lg p-4 shadow">
        <p className="text-xs text-gray-500 uppercase">Gram</p>
        <p className="text-lg font-semibold">{result.gram || '—'}</p>
      </div>
      <div className="bg-white rounded-lg p-4 shadow">
        <p className="text-xs text-gray-500 uppercase">Dominant Shape</p>
        <p className="text-lg font-semibold">{result.dominant_shape || '—'}</p>
      </div>
      <div className="bg-white rounded-lg p-4 shadow">
        <p className="text-xs text-gray-500 uppercase">Colonies</p>
        <p className="text-lg font-semibold">{result.total_colonies ?? '—'}</p>
      </div>
      <div className="bg-white rounded-lg p-4 shadow">
        <p className="text-xs text-gray-500 uppercase">Confidence</p>
        <p className={`text-lg font-bold ${confColor}`}>{conf.toFixed(2)}</p>
      </div>
      <div className="bg-white rounded-lg p-4 shadow">
        <p className="text-xs text-gray-500 uppercase">Organism Type</p>
        <p className="text-lg font-semibold">{result.organism_type || '—'}</p>
      </div>
    </div>
  );
}
