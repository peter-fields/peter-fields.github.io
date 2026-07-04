#!/usr/bin/env python3
"""Escalation: domain search via EBI InterProScan REST API (Pfam + other signatures).

Stable, supported EBI Job Dispatcher service. Async: submit -> poll -> fetch.
A Pfam/InterPro domain hit identifies family membership even when blastp is dark;
a toxin/virulence domain here would be the result that moves the flag decision.
"""

import time
import requests

EMAIL = "pfields97@gmail.com"          # EBI contact field (not a credential)
ROOT = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"

CANDIDATES = {
    "rev_-1": ("MGLVHIRSHKNSGIDNLHAGDCEKGNSNGAVCKTWNTADNKKAVGIVLNSPASNDEYVAI"
               "AVAGPIIVDKLNWGGTDDSEATKKDKFIDLGIMLIH"),
    "fwd_+1": ("MNQHNTQINKFIFFSSFRIISTTPIQFINNNRTSNSNSNIFIIRSRRIQNNTNSFFIISS"
               "IPSFTNSTIRITFFTITSMQIINTRIFMRSNMNQTH"),
}


def submit(label, seq):
    r = requests.post(
        f"{ROOT}/run",
        data={"email": EMAIL, "title": label, "stype": "p",
              "sequence": f">{label}\n{seq}"},
        headers={"Accept": "text/plain"}, timeout=60,
    )
    if r.status_code != 200:
        print(f"  submit failed {r.status_code}: {r.text[:300]}")
        return None
    return r.text.strip()


def wait(jobid, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = requests.get(f"{ROOT}/status/{jobid}",
                         headers={"Accept": "text/plain"}, timeout=30).text.strip()
        if s in ("FINISHED", "ERROR", "FAILURE", "NOT_FOUND"):
            return s
        time.sleep(20)
    return "TIMEOUT"


def show_results(label, jobid):
    r = requests.get(f"{ROOT}/result/{jobid}/json", timeout=120)
    if r.status_code != 200:
        print(f"  result fetch failed {r.status_code}: {r.text[:200]}")
        return
    data = r.json()
    matches = data.get("results", [{}])[0].get("matches", [])
    print(f"  {len(matches)} signature match(es):")
    if not matches:
        print("    (no domain signatures matched)")
        return
    for m in matches:
        sig = m.get("signature", {})
        lib = (sig.get("signatureLibraryRelease") or {}).get("library", "?")
        acc = sig.get("accession", "?")
        desc = sig.get("description") or sig.get("name") or ""
        entry = sig.get("entry") or {}
        ipr = f" | InterPro {entry.get('accession')} {entry.get('name','')} [{entry.get('type','')}]" if entry else ""
        locs = m.get("locations", [])
        ev = locs[0].get("evalue") if locs else None
        sc = locs[0].get("score") if locs else None
        rng = f"{locs[0].get('start')}-{locs[0].get('end')}" if locs else "?"
        print(f"    [{lib}] {acc} {desc}  E={ev} score={sc} aa:{rng}{ipr}")


def main():
    jobs = {}
    for label, seq in CANDIDATES.items():
        jid = submit(label, seq)
        print(f"submitted {label}: {jid}")
        if jid:
            jobs[label] = jid
    for label, jid in jobs.items():
        print(f"\n### {label} ({jid})")
        st = wait(jid)
        print(f"  status: {st}")
        if st == "FINISHED":
            show_results(label, jid)


if __name__ == "__main__":
    main()
