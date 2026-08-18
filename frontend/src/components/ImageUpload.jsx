import { useState, useEffect } from 'react';

export default function ImageUpload({ onResult, onLoading }) {
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);

  useEffect(() => {
    return () => { if (preview) URL.revokeObjectURL(preview); };
  }, [preview]);

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    if (preview) URL.revokeObjectURL(preview);
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (!f) return;
    if (preview) URL.revokeObjectURL(preview);
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const handleAnalyze = async () => {
    if (!file) return;
    onLoading(true);
    const formData = new FormData();
    formData.append('image', file);
    formData.append('mode', 'advanced');
    try {
      const { default: api } = await import('../api');
      const resp = await api.post('/predict', formData);
      onResult(resp.data);
    } catch (err) {
      alert('Prediction failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      onLoading(false);
    }
  };

  return (
    <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
      <div onDrop={handleDrop} onDragOver={(e) => e.preventDefault()} className="min-h-[200px] flex flex-col items-center justify-center gap-4">
        {preview ? (
          <img src={preview} alt="Preview" className="max-h-48 rounded" />
        ) : (
          <p className="text-gray-500">Drag & drop a petri-dish image here</p>
        )}
        <label className="cursor-pointer bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
          Browse
          <input type="file" accept="image/*" onChange={handleFile} className="hidden" />
        </label>
      </div>
      {file && (
        <div className="mt-4">
          <p className="text-sm text-gray-600 mb-2">{file.name}</p>
          <button onClick={handleAnalyze} className="bg-green-600 text-white px-6 py-2 rounded hover:bg-green-700 font-semibold">
            Analyze
          </button>
        </div>
      )}
    </div>
  );
}
