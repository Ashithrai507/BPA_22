from __future__ import annotations


def parse(text: str) -> list[dict]:
    records: list[dict] = []
    header: str | None = None
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append(_record(header, "".join(lines)))
            header = line[1:].strip()
            lines = []
        else:
            lines.append(line)
    if header is not None:
        records.append(_record(header, "".join(lines)))
    return records


def _record(header: str, sequence: str) -> dict:
    first = header.split()[0] if header else ""
    parts = first.split("|")
    accession = parts[1] if len(parts) > 1 and parts[0] in {"sp", "tr"} else parts[0]
    return {"accession": accession, "header": header, "sequence": sequence}


def to_fasta(proteins: list[dict], resolved_name: str) -> str:
    chunks: list[str] = []
    for protein in proteins:
        fields = [
            protein.get("accession"),
            protein.get("gene"),
            protein.get("protein_name"),
            resolved_name,
        ]
        header = "|".join(str(f) for f in fields if f)
        chunks.append(f">{header}\n{_wrap(protein.get("sequence") or "")}")
    return "\n".join(chunks) + ("\n" if chunks else "")


def _wrap(sequence: str, width: int = 60) -> str:
    return "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))
