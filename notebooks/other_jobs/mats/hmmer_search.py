#!/usr/bin/env python3
"""Escalation step: profile / remote-homology search via the EBI HMMER web API.

When plain blastp finds nothing, a profile search is more sensitive:
  - hmmscan vs Pfam : does any region belong to a known domain *family*?
  - phmmer  vs UniProt/SwissProt : sensitive sequence search (per-position model).

We search both ORF candidates. No install/DB download needed -- EBI runs it.
"""

import json
import sys
import requests

CANDIDATES = {
    "+1": ("MNQHNTQINKFIFFSSFRIISTTPIQFINNNRTSNSNSNIFIIRSRRIQNNTNSFFIISS"
           "IPSFTNSTIRITFFTITSMQIINTRIFMRSNMNQTH"),
    "-1": ("MGLVHIRSHKNSGIDNLHAGDCEKGNSNGAVCKTWNTADNKKAVGIVLNSPASNDEYVAI"
           "AVAGPIIVDKLNWGGTDDSEATKKDKFIDLGIMLIH"),
}

BASE = "https://www.ebi.ac.uk/Tools/hmmer/search"


def hmmer(tool: str, dbkey: str, dbval: str, label: str, seq: str):
    """tool: 'hmmscan' (hmmdb) or 'phmmer' (seqdb)."""
    url = f"{BASE}/{tool}"
    fasta = f">{label}\n{seq}\n"
    print(f"\n### {tool} ({dbkey}={dbval}) on candidate {label} ({len(seq)} aa)")
    try:
        r = requests.post(
            url,
            data={dbkey: dbval, "seq": fasta},
            headers={"Accept": "application/json"},
            timeout=180,
            allow_redirects=True,
        )
    except Exception as e:
        print(f"  request error: {type(e).__name__}: {e}")
        return
    print(f"  status={r.status_code}  final_url={r.url}  ctype={r.headers.get('content-type','')}")
    if r.status_code != 200:
        print(f"  body[:300]: {r.text[:300]}")
        return
    try:
        data = r.json()
    except json.JSONDecodeError:
        print(f"  non-JSON response; body[:300]: {r.text[:300]}")
        return

    hits = data.get("results", {}).get("hits", [])
    print(f"  hits: {len(hits)}")
    for h in hits[:15]:
        name = h.get("name", "?")
        desc = (h.get("desc") or "")[:55]
        evalue = h.get("evalue", "?")
        score = h.get("score", "?")
        print(f"    E={evalue:>10}  bits={score:>7}  {name:<14} {desc}")


def main():
    for label, seq in CANDIDATES.items():
        hmmer("hmmscan", "hmmdb", "pfam", label, seq)   # domain-family membership
    # sensitive sequence search on the protein-like reverse-strand candidate
    hmmer("phmmer", "seqdb", "swissprot", "-1", CANDIDATES["-1"])


if __name__ == "__main__":
    main()
