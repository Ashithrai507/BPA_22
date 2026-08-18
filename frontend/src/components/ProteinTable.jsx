export default function ProteinTable({ proteins }) {
  if (!proteins || proteins.length === 0) return null;

  return (
    <table className="min-w-full text-sm bg-white rounded-lg shadow">
      <thead className="bg-gray-100">
        <tr>
          <th className="px-3 py-2 text-left">#</th>
          <th className="px-3 py-2 text-left">Accession</th>
          <th className="px-3 py-2 text-left">Gene</th>
          <th className="px-3 py-2 text-left">Length</th>
          <th className="px-3 py-2 text-left">Score</th>
          <th className="px-3 py-2 text-left">Structure</th>
          <th className="px-3 py-2 text-left">Reviewed</th>
        </tr>
      </thead>
      <tbody>
        {proteins.map((p) => (
          <tr key={p.accession} className="border-t hover:bg-gray-50">
            <td className="px-3 py-2">{p.rank}</td>
            <td className="px-3 py-2 font-mono text-xs">{p.accession}</td>
            <td className="px-3 py-2">{p.gene || '—'}</td>
            <td className="px-3 py-2">{p.length}</td>
            <td className="px-3 py-2 font-semibold">{p.score?.toFixed(2)}</td>
            <td className="px-3 py-2">{p.has_structure ? '✓' : '—'}</td>
            <td className="px-3 py-2">{p.reviewed ? '✓' : '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
