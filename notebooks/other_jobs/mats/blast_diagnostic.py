#!/usr/bin/env python3
"""Diagnostic: relaxed blastp on the reverse-complement candidate, to distinguish
a true 'no significant homolog' from default-stringency / low-complexity masking."""

from Bio.Blast import NCBIWWW, NCBIXML

REV = ("MGLVHIRSHKNSGIDNLHAGDCEKGNSNGAVCKTWNTADNKKAVGIVLNSPASNDEYVAI"
       "AVAGPIIVDKLNWGGTDDSEATKKDKFIDLGIMLIH")

print("blastp -1 candidate vs nr, expect=2000, low-complexity filter OFF ...")
try:
    handle = NCBIWWW.qblast("blastp", "nr", REV, expect=2000.0,
                            hitlist_size=25, filter="F")
    record = NCBIXML.read(handle)
    print(f"alignments returned: {len(record.alignments)}")
    qlen = len(REV)
    for ali in record.alignments[:25]:
        hsp = ali.hsps[0]
        pid = 100.0 * hsp.identities / hsp.align_length
        qcov = 100.0 * (hsp.query_end - hsp.query_start + 1) / qlen
        print(f"  E={hsp.expect:9.1e}  id={pid:3.0f}%  qcov={qcov:3.0f}%  {ali.hit_def[:65]}")
except Exception as e:
    print(f"ERROR from qblast: {type(e).__name__}: {e}")
