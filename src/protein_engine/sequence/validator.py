from __future__ import annotations

from protein_engine.config import MIN_SEQUENCE_LENGTH

AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")


def validate(protein: dict, min_length: int = MIN_SEQUENCE_LENGTH) -> dict:
    issues: list[str] = []
    sequence = protein.get("sequence") or ""
    if len(sequence) < min_length:
        issues.append("too_short")
    non_standard = sorted({residue for residue in sequence.upper() if residue not in AA_ALPHABET})
    if non_standard:
        issues.append(f"non_standard:{''.join(non_standard)}")
    return {"ok": not issues, "protein": protein, "issues": issues}


def dedupe_by_accession(proteins: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []
    for protein in proteins:
        accession = protein.get("accession")
        if accession in seen:
            continue
        seen.add(accession)
        result.append(protein)
    return result


def split_valid(
    proteins: list[dict], min_length: int = MIN_SEQUENCE_LENGTH
) -> tuple[list[dict], list[dict]]:
    valid: list[dict] = []
    excluded: list[dict] = []
    for protein in dedupe_by_accession(proteins):
        check = validate(protein, min_length)
        if check["ok"]:
            valid.append(protein)
        else:
            excluded.append({**protein, "excluded_reasons": check["issues"]})
    return valid, excluded
