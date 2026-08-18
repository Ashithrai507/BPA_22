import { useState, useEffect } from 'react';
import api from '../api';

export default function SpeciesSelector({ value, onChange }) {
  const [species, setSpecies] = useState([]);

  useEffect(() => {
    api.get('/species').then((r) => setSpecies(r.data)).catch(() => {});
  }, []);

  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className="border rounded px-3 py-2 text-sm">
      <option value="">Select species...</option>
      {species.map((s) => (
        <option key={s.name} value={s.name}>{s.name} ({s.gram})</option>
      ))}
    </select>
  );
}
