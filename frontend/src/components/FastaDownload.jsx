export default function FastaDownload({ proteins, species }) {
  if (!proteins || proteins.length === 0) return null;

  const fasta = proteins.map((p) => `>${p.accession}|${species}|${p.gene || 'unknown'}|${p.length}\n${p.sequence || ''}`).join('\n');

  const download = () => {
    const blob = new Blob([fasta], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${species.replace(/\s+/g, '_')}_proteins.fasta`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <button onClick={download} className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 text-sm">
      Download FASTA
    </button>
  );
}
