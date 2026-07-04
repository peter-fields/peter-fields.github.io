#!/usr/bin/env python3
"""
DNA sequence screening demo (MATS / Gary Abel stream, Part 2).

Implements Part-1 steps 1-2: take a DNA sequence, translate all six reading
frames, extract the longest ORF, and blastp it against a protein database to
identify the nearest known homolog.

Rationale: the company's *nucleotide* screen returned no close matches. If the
order was obfuscated by synonymous-codon substitution, the DNA looks novel while
the encoded *protein* is unchanged -- so we search at the amino-acid level.

Usage:
    python screen_sequence.py                 # translate + find ORF only (instant, offline)
    python screen_sequence.py --blast         # also blastp the ORF (sends sequence to NCBI)
    python screen_sequence.py --blast --db swissprot   # curated DB (default: nr)
    python screen_sequence.py --blast --db nr --expect 2000 --no-filter --hitlist 25
                                              # relaxed search: surface weak/chance hits

Requires: biopython  (pip install biopython)
"""

import argparse
import textwrap

from Bio.Seq import Seq

# The mystery order under review.
MYSTERY = (
    "ATGAATCAACATAATACCCAAATCAATAAATTTATCTTTTTTAGTAGCTTCAGAATCATCAGTACC"
    "ACCCCAATTCAATTTATCAACAATAATAGGACCAGCAACAGCAATAGCAACATATTCATCATTAGA"
    "AGCAGGAGAATTCAAAACAATACCAACAGCTTTTTTATTATCAGCAGTATTCCAAGTTTTACAAAC"
    "AGCACCATTAGAATTACCTTTTTCACAATCACCAGCATGCAAATTATCAATACCAGAATTTTTATG"
    "AGATCTAATATGAACCAAACCCAT"
)


def six_frame_translations(dna: str):
    """Return {frame_label: protein_str} for all six reading frames.

    Frames +1/+2/+3 read the given strand 5'->3'; frames -1/-2/-3 read the
    reverse complement 5'->3' (the other physical strand). A protein can only be
    encoded 5'->3' along a strand, so these six frames are the complete set.
    """
    seq = Seq(dna)
    frames = {}
    for strand, nuc in [("+", seq), ("-", seq.reverse_complement())]:
        for offset in range(3):
            label = f"{strand}{offset + 1}"
            # Trim to a multiple of 3 so Biopython doesn't warn on a partial codon.
            sub = nuc[offset:]
            sub = sub[: len(sub) - (len(sub) % 3)]
            frames[label] = str(sub.translate())
    return frames


def longest_orf(protein: str):
    """Longest ORF starting at Met in a protein string. Returns (peptide, start_idx).

    peptide includes the leading Met and excludes the stop codon. If a Met has no
    in-frame stop downstream, the ORF runs to the end of the frame -- this is the
    common case for a CDS-only insert (no stop codon synthesized).
    """
    best_pep, best_start = None, None
    i = 0
    while i < len(protein):
        if protein[i] == "M":
            stop = protein.find("*", i)
            pep = protein[i:] if stop == -1 else protein[i:stop]
            if best_pep is None or len(pep) > len(best_pep):
                best_pep, best_start = pep, i
            i = (i + 1) if stop == -1 else (stop + 1)
        else:
            i += 1
    return best_pep, best_start


def best_orf_per_strand(dna: str):
    """Return the best ORF candidate on each strand: {'+': (label, pep), '-': (label, pep)}.

    We keep one candidate per strand rather than a single global best, because the
    given strand and its reverse complement can *both* present a clean ORF -- and
    which one is the real gene is exactly what the homology search resolves.
    """
    best = {"+": (None, None, -1), "-": (None, None, -1)}
    for label, prot in six_frame_translations(dna).items():
        strand = label[0]
        pep, _ = longest_orf(prot)
        if pep and len(pep) > best[strand][2]:
            best[strand] = (label, pep, len(pep))
    return {s: (lbl, pep) for s, (lbl, pep, _) in best.items()}


def run_blast(peptide: str, db: str, expect: float = 10.0,
              no_filter: bool = False, hitlist: int = 10):
    """blastp the peptide against an NCBI database; print the top hits.

    expect     : E-value reporting threshold. Raise it (e.g. 2000) to surface weak
                 / likely-chance hits when the default (10) returns nothing.
    no_filter  : if True, disable the SEG low-complexity filter (qblast filter='F').
    hitlist    : max number of hits to retrieve.
    """
    from Bio.Blast import NCBIWWW, NCBIXML

    kwargs = dict(expect=expect, hitlist_size=hitlist)
    if no_filter:
        kwargs["filter"] = "F"  # disable SEG low-complexity masking of the query
    print(f"\nblastp vs {db}, expect={expect:g}, low-complexity filter "
          f"{'OFF' if no_filter else 'default'} (this can take a few minutes)...")
    handle = NCBIWWW.qblast("blastp", db, peptide, **kwargs)
    record = NCBIXML.read(handle)

    if not record.alignments:
        print("No hits returned.")
        return

    qlen = len(peptide)
    print(f"alignments returned: {len(record.alignments)}")
    print(f"{'E-value':>10}  {'%id':>5}  {'qcov':>5}  description")
    print("-" * 80)
    for ali in record.alignments[:hitlist]:
        hsp = ali.hsps[0]  # best HSP for this hit
        pct_id = 100.0 * hsp.identities / hsp.align_length
        qcov = 100.0 * (hsp.query_end - hsp.query_start + 1) / qlen
        desc = ali.hit_def.split(">")[0][:60]
        print(f"{hsp.expect:10.1e}  {pct_id:5.0f}  {qcov:5.0f}  {desc}")


def main():
    ap = argparse.ArgumentParser(description="DNA screening demo: 6-frame translate + ORF + blastp")
    ap.add_argument("--seq", default=MYSTERY, help="DNA sequence (default: the mystery order)")
    ap.add_argument("--blast", action="store_true", help="also run blastp (sends sequence to NCBI)")
    ap.add_argument("--db", default="nr", help="BLAST database: nr (default) or swissprot")
    ap.add_argument("--expect", type=float, default=10.0, help="E-value reporting threshold (default 10)")
    ap.add_argument("--no-filter", action="store_true", help="disable the SEG low-complexity filter")
    ap.add_argument("--hitlist", type=int, default=10, help="max hits to retrieve (default 10)")
    args = ap.parse_args()

    dna = args.seq.strip().upper()
    print(f"Input DNA: {len(dna)} nt\n")

    print("Six-frame translation (stop codons per frame):")
    for label, prot in six_frame_translations(dna).items():
        pep, _ = longest_orf(prot)
        n = len(pep) if pep else 0
        print(f"  frame {label}: {prot.count('*'):2d} stop(s), longest ORF = {n:3d} aa")

    candidates = best_orf_per_strand(dna)
    print("\nORF candidates (one per strand):")
    for strand in ("+", "-"):
        label, pep = candidates[strand]
        if pep:
            print(f"\n  [{label}] {len(pep)} aa")
            print("\n".join("    " + line for line in textwrap.wrap(pep, 60)))

    if args.blast:
        for strand in ("+", "-"):
            label, pep = candidates[strand]
            if pep:
                print(f"\n{'=' * 80}\nBLAST candidate from frame {label} ({len(pep)} aa)")
                run_blast(pep, args.db, expect=args.expect,
                          no_filter=args.no_filter, hitlist=args.hitlist)
    else:
        print("\n(Re-run with --blast to identify the proteins via NCBI blastp.)")


if __name__ == "__main__":
    main()
