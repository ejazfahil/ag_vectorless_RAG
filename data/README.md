# Data Directory

## Structure
- `raw/` — Original documents (PDF, DOCX, TXT, MD). **Gitignored.**
- `processed/` — Preprocessed JSON with hierarchical structure. Committed.
- `golden_qa/` — Ground-truth Q&A pairs for evaluation. Committed.

## Domains
| Domain | Description | Target Docs |
|--------|-------------|-------------|
| `finance/` | SEC filings, earnings reports (FinanceBench) | ~50 |
| `legal/` | Contracts, legal clauses (CUAD dataset) | ~50 |
| `technical/` | ArXiv papers, product manuals | ~50 |

## Adding Documents
1. Place raw documents in `data/raw/{domain}/`
2. Run: `python -c "from src.corpus import CorpusPreprocessor; p=CorpusPreprocessor(); c=p.process('data/raw/finance/', 'finance'); p.save(c, 'data/processed/finance/')"`
3. Generate Q&A: `python -c "from src.corpus import GoldenQAGenerator; ..."`
