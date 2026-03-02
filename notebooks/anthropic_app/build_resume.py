#!/usr/bin/env python3
"""
Build Fields_Peter_Resume_final.docx from the updated .docx template
(preserving styles, margins, numbering) with new content from the .md.
"""
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

ORIG = '/Users/pfields/Git/peter-fields.github.io/notebooks/anthropic_app/Fields_Peter_Resume_g_updated.docx'
OUT  = '/Users/pfields/Git/peter-fields.github.io/notebooks/anthropic_app/Fields_Peter_Resume_final.docx'

doc  = Document(ORIG)
body = doc.element.body

# ── Extract bullet pPr template (numPr + spacing + indent) ──────────────────
bullet_pPr_tpl = None
for para in doc.paragraphs:
    pPr_el = para._element.find(qn('w:pPr'))
    if pPr_el is not None and pPr_el.find(qn('w:numPr')) is not None:
        bullet_pPr_tpl = copy.deepcopy(pPr_el)
        break

# ── Clear all paragraphs (keep sectPr) ──────────────────────────────────────
for p in list(body.findall(qn('w:p'))):
    body.remove(p)

# ── Low-level builders ───────────────────────────────────────────────────────
XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'

def _run(text, bold=None, italic=None, size=None):
    r = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    if bold is True:
        rPr.append(OxmlElement('w:b'))
        rPr.append(OxmlElement('w:bCs'))
    if italic is True:
        rPr.append(OxmlElement('w:i'))
        rPr.append(OxmlElement('w:iCs'))
    if size is not None:
        for tag in ('w:sz', 'w:szCs'):
            el = OxmlElement(tag)
            el.set(qn('w:val'), str(int(size * 2)))
            rPr.append(el)
    if len(rPr):
        r.append(rPr)
    t = OxmlElement('w:t')
    t.text = text
    if text and (text[0] == ' ' or text[-1] == ' '):
        t.set(XML_SPACE, 'preserve')
    r.append(t)
    return r

def _pPr(sp_before=None, sp_after=None):
    pPr = OxmlElement('w:pPr')
    if sp_before is not None or sp_after is not None:
        spacing = OxmlElement('w:spacing')
        if sp_before is not None:
            spacing.set(qn('w:before'), str(int(sp_before * 20)))
        if sp_after is not None:
            spacing.set(qn('w:after'), str(int(sp_after * 20)))
        pPr.append(spacing)
    return pPr

def _emit(pPr_elem, runs):
    p = OxmlElement('w:p')
    p.append(pPr_elem)
    for r in runs:
        p.append(r)
    sectPr = body.find(qn('w:sectPr'))
    if sectPr is not None:
        sectPr.addprevious(p)
    else:
        body.append(p)
    return p

# ── Content helpers ──────────────────────────────────────────────────────────

def NAME(text):
    _emit(_pPr(sp_after=2), [_run(text, size=20)])

def TAGLINE(text):
    _emit(_pPr(), [_run(text)])

def CONTACT(text):
    _emit(_pPr(), [_run(text, size=9)])

def SECTION(text):
    _emit(_pPr(sp_before=10, sp_after=4), [_run(text, bold=True, size=11)])

def TITLE(text, italic=False):
    _emit(_pPr(sp_after=1), [_run(text, bold=True, italic=italic or None, size=10)])

def BODY(text, italic=False, size=9):
    _emit(_pPr(sp_after=1), [_run(text, italic=italic or None, size=size)])

def MIXED(*runs_data):
    """Each item: (text, bold, italic, size) — use None to inherit."""
    _emit(_pPr(sp_after=1), [_run(t, b, i, s) for t, b, i, s in runs_data])

def BULLET(text):
    _emit(copy.deepcopy(bullet_pPr_tpl), [_run(text, size=9)])

def BLANK():
    _emit(_pPr(sp_after=1), [])


# ════════════════════════════════════════════════════════════════════════════
#  DOCUMENT CONTENT
# ════════════════════════════════════════════════════════════════════════════

NAME('Peter W. Fields')
TAGLINE('Physicist | Machine Learning Researcher')
CONTACT('Chicago, IL • (646) 599-5151')
CONTACT('pfields97@gmail.com  •  peter-fields.github.io  •  github.com/peter-fields  •  Google Scholar')

# ── SUMMARY ──────────────────────────────────────────────────────────────────
SECTION('SUMMARY')
BULLET('PhD researcher in statistical physics / machine learning with 6 years of experience '
       'building and evaluating probabilistic models (energy-based / graphical models) '
       'grounded in theory and validated empirically.')
BULLET('Recently developed forward-pass-only diagnostic tools for mechanistic interpretability '
       'of transformer attention heads; validated on the IOI circuit in GPT-2.')
BULLET('End-to-end modeling workflow: problem framing → rapid prototyping → metric design & '
       'evaluation (held-out performance, ablations) → reproducible research software and documentation.')

# ── EDUCATION ────────────────────────────────────────────────────────────────
SECTION('EDUCATION')
TITLE('PhD in Physics, University of Chicago – March, 2026')
BODY('Dissertation submitted; defense completed', italic=True)
BODY('Advisor: Stephanie E. Palmer')
BLANK()
TITLE('Bachelor of Science in Physics, The City College of New York – 2019')

# ── PUBLICATIONS & PREPRINTS ─────────────────────────────────────────────────
SECTION('PUBLICATIONS & PREPRINTS')

TITLE('Understanding temperature tuning in energy-based models')
BODY('P. W. Fields, V. Ngampruetikorn, D. J. Schwab, S. E. Palmer.')
BODY('Preprint (2025). arXiv:2512.09152')
BODY('Code: github.com/peter-fields/temp-tune')
BLANK()

TITLE('Understanding Energy-Based Modeling of Proteins via an Empirically Motivated '
      'Minimal Ground Truth Model')
BODY('P. W. Fields, V. Ngampruetikorn, R. Ranganathan, D. J. Schwab, S. E. Palmer.')
BODY('Synergy of Scientific and Machine Learning Modeling Workshop at '
     'International Conference in Machine Learning (2023).')
BODY('Presented at ICML 2023 workshop.', italic=True)
BODY('Code: github.com/peter-fields/toysector')
BLANK()

TITLE('Tunable Pseudocapacitive Intercalation of Chloroaluminate Anions into Graphite '
      'Electrodes for Rechargeable Aluminum Batteries')
MIXED(
    ('J. H. Xu, T. Schoetz, J. R. McManus, V. R. Subramanian, ', None, None, 9),
    ('P. W. Fields', True, None, 9),
    (', R. J. Messinger.', None, None, 9),
)
BODY('Journal of the Electrochemical Society 168, 060514 (2021)', italic=True)

# ── SELECTED WRITING ─────────────────────────────────────────────────────────
SECTION('SELECTED WRITING')
TITLE('Attention Diagnostics: Testing KL and Susceptibility on the IOI Circuit')
BODY('peter-fields.github.io/attention-diagnostics/')
BLANK()
TITLE('Why Softmax? A Hypothesis Testing Perspective on Attention Weights')
BODY('peter-fields.github.io/why-softmax/')

# ── EXPERIENCE ───────────────────────────────────────────────────────────────
SECTION('EXPERIENCE')

TITLE('Palmer Theory Group, University of Chicago (Jan 2021 – Present)')
BULLET('Data-driven modeling of proteins and neural-like systems using statistical physics '
       'and machine learning methods')
BULLET('Utilized energy-based models to gain insight into biological function')
BULLET('Developed synthetic benchmarks to study finite-sample effects and ground truth structure')
BULLET('Developed forward-pass diagnostic tools for mechanistic interpretability of transformer '
       'attention heads; demonstrated statistically significant (p<0.001) distinguishability '
       'between circuit and non-circuit heads in GPT-2\'s IOI circuit')
BLANK()

TITLE('Messinger Research Group, The City College of New York (Jan 2018 – Aug 2019)')
BULLET('Physics-based characterization of novel battery technologies')
BULLET('Designed and executed impedance spectroscopy experiments')
BULLET('Studied diffusive, kinetic, and electrical properties of battery systems')
BLANK()

TITLE('Teaching Assistant, General Physics I & II, University of Chicago (2019–2021)')

# ── SKILLS ───────────────────────────────────────────────────────────────────
SECTION('SKILLS')
MIXED(
    ('Coding: ', True, None, None),
    ('Julia (expert), Python (PyTorch), Mathematica, MATLAB, Unix/CLI, LaTeX, Git, Jupyter',
     None, None, None),
)
MIXED(('Methods:', True, None, None))
BULLET('Information theory; statistical inference; regularization; hyperparameter search; model selection')
BULLET('Supervised and generative/probabilistic models')
BULLET('Graphical models; network/graph structure inference')

# ── SELECTED TALKS & PRESENTATIONS ───────────────────────────────────────────
SECTION('SELECTED TALKS & PRESENTATIONS')
TITLE('Temperature-tuning trained energy functions improves generative performance', italic=True)
BODY('APS Global Physics Summit 2025 – P. W. Fields, V. Ngampruetikorn, D. J. Schwab, S. E. Palmer')
BLANK()
TITLE('Characterizing Inference of Non-reciprocal Connections in the Kinetic Ising Model', italic=True)
BODY('APS March Meeting 2024 – P. W. Fields, Cheyne Weis, S. E. Palmer, Peter Littlewood')

# ── AWARDS ────────────────────────────────────────────────────────────────────
SECTION('AWARDS')
TITLE('Soft and Living Matter Exchange Award, Institute for Complex Adaptive Matter (Dec 2023)')
BLANK()
TITLE('Center for Physics of Evolving Systems Fellowship, University of Chicago (Jan 2022–Dec 2023)')

# ════════════════════════════════════════════════════════════════════════════
doc.save(OUT)
print(f'Saved → {OUT}')