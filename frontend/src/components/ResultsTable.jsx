export default function ResultsTable({ colonies }) {
  if (!colonies || colonies.length === 0) return null;

  return (
    <div className="mt-4 overflow-x-auto">
      <h3 className="text-sm font-semibold text-gray-700 mb-2">Colony Measurements ({colonies.length} detected)</h3>
      <table className="min-w-full text-sm bg-white rounded-lg shadow">
        <thead className="bg-gray-100">
          <tr>
            <th className="px-3 py-2 text-left">#</th>
            <th className="px-3 py-2 text-left">Area</th>
            <th className="px-3 py-2 text-left">Perimeter</th>
            <th className="px-3 py-2 text-left">Circularity</th>
            <th className="px-3 py-2 text-left">Aspect Ratio</th>
            <th className="px-3 py-2 text-left">Solidity</th>
            <th className="px-3 py-2 text-left">Shape</th>
          </tr>
        </thead>
        <tbody>
          {colonies.map((c) => (
            <tr key={c.id} className="border-t hover:bg-gray-50">
              <td className="px-3 py-2">{c.id}</td>
              <td className="px-3 py-2">{c.area?.toFixed(1)}</td>
              <td className="px-3 py-2">{c.perimeter?.toFixed(1)}</td>
              <td className="px-3 py-2">{c.circularity?.toFixed(2)}</td>
              <td className="px-3 py-2">{c.aspect_ratio?.toFixed(2)}</td>
              <td className="px-3 py-2">{c.solidity?.toFixed(2)}</td>
              <td className="px-3 py-2">{c.predicted_shape}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
