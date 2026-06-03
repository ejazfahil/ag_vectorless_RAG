#!/usr/bin/env python3
"""
Dataset Downloader — fetches FinanceBench, CUAD, and technical docs
for the Vectorless RAG Benchmark.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --domain finance
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_RAW = PROJECT_ROOT / "data" / "raw"


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress indication."""
    print(f"  ⬇ Downloading {desc or url}...")
    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved: {dest.name} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_financebench():
    """
    Download FinanceBench dataset from PatronusAI GitHub.
    Contains 150 Q&A pairs over SEC filings with evidence strings.
    """
    print("\n📊 FinanceBench (Finance Domain)")
    print("=" * 50)

    finance_dir = DATA_RAW / "finance"
    finance_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download the main dataset JSON (Q&A pairs + evidence)
    qa_url = "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl"
    download_file(qa_url, finance_dir / "financebench_qa.jsonl", "FinanceBench Q&A (JSONL)")

    # 2. Also try the CSV format
    csv_url = "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.csv"
    download_file(csv_url, finance_dir / "financebench_qa.csv", "FinanceBench Q&A (CSV)")

    # 3. Download sample SEC filing PDFs for actual RAG testing
    # These are publicly available from SEC EDGAR

    # Try to get file list from the repo
    docs_index_url = "https://api.github.com/repos/patronus-ai/financebench/contents/data"
    try:
        req = urllib.request.Request(docs_index_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            contents = json.loads(resp.read())
            # Find any downloadable docs
            for item in contents:
                if item.get("type") == "dir" and item.get("name") == "docs":
                    print(f"  ℹ Found docs directory at: {item['html_url']}")
    except Exception:
        pass

    # Create sample finance documents from the Q&A data itself
    _create_finance_docs_from_qa(finance_dir)

    print(f"\n  ✅ Finance data ready in: {finance_dir}")


def _create_finance_docs_from_qa(finance_dir: Path):
    """
    Create structured text documents from FinanceBench Q&A evidence strings.
    This gives us real financial text to run RAG against.
    """
    qa_file = finance_dir / "financebench_qa.jsonl"
    if not qa_file.exists():
        # Try CSV
        qa_file = finance_dir / "financebench_qa.csv"
        if not qa_file.exists():
            print("  ⚠ No Q&A file found, creating synthetic finance docs...")
            _create_synthetic_finance_docs(finance_dir)
            return

    docs_dir = finance_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    try:
        # Parse JSONL
        entries_by_company = {}
        with open(finance_dir / "financebench_qa.jsonl", "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    company = entry.get("company", entry.get("ticker", "unknown"))
                    if company not in entries_by_company:
                        entries_by_company[company] = []
                    entries_by_company[company].append(entry)
                except json.JSONDecodeError:
                    continue

        # Create one document per company
        for company, entries in list(entries_by_company.items())[:20]:
            doc_path = docs_dir / f"{company}_financial_data.txt"
            with open(doc_path, "w") as f:
                f.write(f"# {company} — Financial Information\n\n")
                for i, entry in enumerate(entries):
                    evidence = entry.get("evidence", entry.get("evidence_text", ""))
                    question = entry.get("question", "")
                    entry.get("answer", "")
                    f.write(f"## Section {i+1}: {question[:80]}\n\n")
                    f.write(f"{evidence}\n\n")
                    f.write("---\n\n")

            print(f"  ✓ Created: {doc_path.name} ({len(entries)} sections)")

    except Exception as e:
        print(f"  ⚠ Error parsing Q&A: {e}, creating synthetic docs...")
        _create_synthetic_finance_docs(finance_dir)


def _create_synthetic_finance_docs(finance_dir: Path):
    """Create realistic synthetic finance docs for testing."""
    docs_dir = finance_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    companies = [
        ("Apple Inc", "AAPL", "Technology"),
        ("Amazon.com Inc", "AMZN", "E-Commerce/Cloud"),
        ("Microsoft Corp", "MSFT", "Technology/Cloud"),
        ("JPMorgan Chase", "JPM", "Banking"),
        ("Johnson & Johnson", "JNJ", "Healthcare"),
    ]

    for name, ticker, sector in companies:
        doc_path = docs_dir / f"{ticker}_10K_2024.txt"
        with open(doc_path, "w") as f:
            f.write(f"""# {name} ({ticker}) — Annual Report 10-K FY2024

## PART I

### Item 1. Business Overview

{name} is a leading company in the {sector} sector. The company operates
across multiple geographic regions including North America, Europe, and
Asia-Pacific. Total headcount as of December 31, 2024 was approximately
{50000 + hash(ticker) % 200000:,} employees.

### Item 1A. Risk Factors

The company faces risks including market competition, regulatory changes,
cybersecurity threats, and macroeconomic conditions. Supply chain disruptions
and geopolitical tensions remain significant concerns for FY2025.

## PART II

### Item 6. Selected Financial Data

| Metric | FY2024 | FY2023 | FY2022 |
|--------|--------|--------|--------|
| Revenue ($B) | {100 + hash(ticker) % 300:.1f} | {95 + hash(ticker) % 280:.1f} | {90 + hash(ticker) % 260:.1f} |
| Net Income ($B) | {20 + hash(ticker) % 60:.1f} | {18 + hash(ticker) % 55:.1f} | {16 + hash(ticker) % 50:.1f} |
| EPS ($) | {5 + hash(ticker) % 15:.2f} | {4.5 + hash(ticker) % 14:.2f} | {4 + hash(ticker) % 13:.2f} |
| Total Assets ($B) | {200 + hash(ticker) % 400:.1f} | {190 + hash(ticker) % 380:.1f} | {180 + hash(ticker) % 360:.1f} |

### Item 7. Management's Discussion and Analysis

Revenue for fiscal year 2024 increased by approximately 5.3% compared to
the prior year, driven primarily by growth in cloud services and digital
transformation initiatives. Operating margins improved by 120 basis points
to {25 + hash(ticker) % 15:.1f}% due to cost optimization programs implemented
in Q2 2024.

The company's gross margin was {55 + hash(ticker) % 20:.1f}%, reflecting
improved supply chain efficiency and favorable product mix.

### Item 8. Financial Statements

#### Balance Sheet Summary
Total current assets: ${50 + hash(ticker) % 100:.1f}B
Total non-current assets: ${150 + hash(ticker) % 300:.1f}B
Total current liabilities: ${40 + hash(ticker) % 80:.1f}B
Long-term debt: ${30 + hash(ticker) % 60:.1f}B
Stockholders' equity: ${100 + hash(ticker) % 200:.1f}B

#### Cash Flow Statement
Operating cash flow: ${25 + hash(ticker) % 50:.1f}B
Capital expenditures: ${10 + hash(ticker) % 20:.1f}B
Free cash flow: ${15 + hash(ticker) % 35:.1f}B

## PART III

### Item 10. Directors and Corporate Governance

The Board of Directors consists of 12 members, with 10 independent directors.
The Audit Committee met 8 times during FY2024.

## PART IV

### Item 15. Exhibits and Financial Statement Schedules

All financial statements have been audited by the company's independent
registered public accounting firm.
""")
        print(f"  ✓ Created: {doc_path.name}")


def download_cuad():
    """
    Download CUAD (Contract Understanding Atticus Dataset).
    Contains 510 legal contracts with 13K+ clause annotations.
    """
    print("\n⚖️ CUAD (Legal Domain)")
    print("=" * 50)

    legal_dir = DATA_RAW / "legal"
    legal_dir.mkdir(parents=True, exist_ok=True)

    # Download CUAD from the Atticus Project (Zenodo/GitHub)
    cuad_json_url = "https://raw.githubusercontent.com/TheAtticusProject/cuad/main/data/CUAD_v1/CUADv1.json"
    download_file(cuad_json_url, legal_dir / "CUADv1.json", "CUAD Q&A annotations")

    # Create structured legal documents from CUAD data
    _create_legal_docs_from_cuad(legal_dir)

    print(f"\n  ✅ Legal data ready in: {legal_dir}")


def _create_legal_docs_from_cuad(legal_dir: Path):
    """Extract and create documents from CUAD JSON."""
    cuad_path = legal_dir / "CUADv1.json"
    docs_dir = legal_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    if cuad_path.exists():
        try:
            with open(cuad_path, "r") as f:
                cuad_data = json.load(f)

            # CUAD is in SQuAD format: {"data": [{"title": ..., "paragraphs": [...]}]}
            articles = cuad_data.get("data", [])
            for article in articles[:20]:  # First 20 contracts
                title = article.get("title", "unknown_contract")
                safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)[:60]
                doc_path = docs_dir / f"{safe_title}.txt"

                with open(doc_path, "w") as f:
                    f.write(f"# {title}\n\n")
                    for para in article.get("paragraphs", []):
                        context = para.get("context", "")
                        f.write(f"{context}\n\n---\n\n")

                size_kb = doc_path.stat().st_size / 1024
                print(f"  ✓ Created: {safe_title}.txt ({size_kb:.1f} KB)")

        except Exception as e:
            print(f"  ⚠ Error parsing CUAD: {e}, creating synthetic docs...")
            _create_synthetic_legal_docs(legal_dir)
    else:
        print("  ⚠ CUAD download failed, creating synthetic legal docs...")
        _create_synthetic_legal_docs(legal_dir)


def _create_synthetic_legal_docs(legal_dir: Path):
    """Create realistic synthetic legal contracts."""
    docs_dir = legal_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    contracts = [
        "Software_License_Agreement",
        "Employment_Agreement",
        "Non_Disclosure_Agreement",
        "Master_Services_Agreement",
        "Asset_Purchase_Agreement",
    ]

    for contract_name in contracts:
        doc_path = docs_dir / f"{contract_name}.txt"
        with open(doc_path, "w") as f:
            f.write(f"""# {contract_name.replace('_', ' ')}

## 1. DEFINITIONS

1.1 "Agreement" means this {contract_name.replace('_', ' ')} including all
exhibits, schedules, and amendments thereto.

1.2 "Confidential Information" means all non-public information disclosed
by either party that is designated as confidential or that reasonably
should be considered confidential.

1.3 "Effective Date" means the date first written above.

## 2. SCOPE OF AGREEMENT

2.1 This Agreement governs the terms and conditions under which the parties
shall conduct business as described herein.

2.2 The scope of services shall be as set forth in Exhibit A attached hereto.

## 3. TERM AND TERMINATION

3.1 The initial term shall be two (2) years from the Effective Date.

3.2 Either party may terminate this Agreement for convenience upon ninety
(90) days' prior written notice.

3.3 Either party may terminate for cause if the other party materially
breaches this Agreement and fails to cure such breach within thirty (30)
days after written notice.

## 4. COMPENSATION AND PAYMENT

4.1 Fees shall be as set forth in Exhibit B.

4.2 Payment terms are Net 30 from the date of invoice.

4.3 Late payments shall accrue interest at 1.5% per month.

## 5. INTELLECTUAL PROPERTY

5.1 Each party retains all rights in its pre-existing intellectual property.

5.2 Work product created under this Agreement shall be owned by the
commissioning party, subject to Section 5.3.

5.3 The provider retains a non-exclusive license to any general tools,
methodologies, or know-how developed during performance.

## 6. LIMITATION OF LIABILITY

6.1 EXCEPT FOR BREACHES OF CONFIDENTIALITY, NEITHER PARTY SHALL BE LIABLE
FOR INDIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES.

6.2 Each party's total aggregate liability shall not exceed the total fees
paid or payable in the twelve (12) months preceding the claim.

## 7. INDEMNIFICATION

7.1 Each party shall indemnify and hold harmless the other party from and
against any claims arising from the indemnifying party's breach of this
Agreement or negligent acts.

## 8. GOVERNING LAW

8.1 This Agreement shall be governed by and construed in accordance with
the laws of the State of Delaware.

8.2 Any disputes shall be resolved through binding arbitration in accordance
with the rules of the American Arbitration Association.

## 9. MISCELLANEOUS

9.1 This Agreement constitutes the entire agreement between the parties.

9.2 No modification shall be effective unless in writing and signed by
both parties.

9.3 If any provision is found to be unenforceable, the remaining provisions
shall continue in full force and effect.
""")
        print(f"  ✓ Created: {contract_name}.txt")


def download_technical():
    """
    Download/create technical documentation corpus.
    Uses a mix of real open-source docs and structured technical content.
    """
    print("\n🔧 Technical Documentation Domain")
    print("=" * 50)

    tech_dir = DATA_RAW / "technical"
    tech_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = tech_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    # Download real open-source documentation (Markdown)
    docs_to_fetch = [
        {
            "name": "Python_argparse_docs.txt",
            "url": "https://raw.githubusercontent.com/python/cpython/main/Doc/library/argparse.rst",
            "desc": "Python argparse documentation",
        },
        {
            "name": "Python_logging_docs.txt",
            "url": "https://raw.githubusercontent.com/python/cpython/main/Doc/library/logging.rst",
            "desc": "Python logging documentation",
        },
        {
            "name": "Python_json_docs.txt",
            "url": "https://raw.githubusercontent.com/python/cpython/main/Doc/library/json.rst",
            "desc": "Python json documentation",
        },
    ]

    for doc in docs_to_fetch:
        dest = docs_dir / doc["name"]
        download_file(doc["url"], dest, doc["desc"])

    # Create additional structured technical manuals
    _create_synthetic_technical_docs(docs_dir)

    print(f"\n  ✅ Technical data ready in: {tech_dir}")


def _create_synthetic_technical_docs(docs_dir: Path):
    """Create structured technical documentation."""
    manuals = {
        "API_Reference_Manual.txt": """# REST API Reference Manual v3.2

## 1. Authentication

### 1.1 API Key Authentication
All API requests must include an API key in the Authorization header:
```
Authorization: Bearer sk-your-api-key-here
```

### 1.2 OAuth 2.0
For user-delegated access, use the OAuth 2.0 authorization code flow.
Redirect URI must be pre-registered in the developer console.

### 1.3 Rate Limits
- Free tier: 100 requests/minute
- Pro tier: 1,000 requests/minute
- Enterprise: Custom (contact sales)

Rate limit headers returned: X-RateLimit-Limit, X-RateLimit-Remaining

## 2. Endpoints

### 2.1 GET /api/v3/documents
Retrieve a list of documents.

Parameters:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| page | int | No | Page number (default: 1) |
| limit | int | No | Results per page (max: 100) |
| sort | string | No | Sort field (created_at, name) |
| filter | string | No | Filter expression |

Response: 200 OK
```json
{
  "data": [...],
  "pagination": {"page": 1, "total_pages": 10, "total_items": 95}
}
```

### 2.2 POST /api/v3/documents
Create a new document.

Request Body:
```json
{
  "title": "My Document",
  "content": "Document content here",
  "tags": ["finance", "q3-report"]
}
```

### 2.3 PUT /api/v3/documents/{id}
Update an existing document. Requires document ownership or admin role.

### 2.4 DELETE /api/v3/documents/{id}
Soft-delete a document. Can be restored within 30 days.

## 3. Error Handling

### 3.1 Error Codes
| Code | Description | Action |
|------|-------------|--------|
| 400 | Bad Request | Check request parameters |
| 401 | Unauthorized | Verify API key |
| 403 | Forbidden | Check permissions |
| 404 | Not Found | Resource doesn't exist |
| 429 | Rate Limited | Retry after X-Retry-After seconds |
| 500 | Internal Error | Contact support |

### 3.2 Error Response Format
```json
{
  "error": {
    "code": "RATE_LIMITED",
    "message": "Too many requests",
    "retry_after": 30
  }
}
```

## 4. Webhooks

### 4.1 Configuration
Register webhook URLs in Settings > Integrations > Webhooks.

### 4.2 Events
- document.created
- document.updated
- document.deleted
- user.invited

### 4.3 Verification
All webhook payloads include an X-Signature header for HMAC-SHA256 verification.

## 5. SDKs and Libraries

Official SDKs: Python, JavaScript/TypeScript, Go, Java
Community SDKs: Ruby, PHP, Rust, C#
""",
        "Database_Admin_Guide.txt": """# Database Administration Guide — PostgreSQL Cluster

## 1. Architecture Overview

### 1.1 Cluster Topology
Primary-Replica architecture with automatic failover via Patroni.
- Primary: Handles all writes and real-time reads
- Replicas (2): Handle read-only queries with streaming replication
- PgBouncer: Connection pooling (max 500 connections)

### 1.2 Storage Configuration
- Data directory: /var/lib/postgresql/16/main
- WAL directory: /var/lib/postgresql/16/wal (separate SSD)
- Tablespace: ssd_fast for hot data, hdd_archive for cold data

## 2. Backup and Recovery

### 2.1 Backup Strategy
- Full backup: Weekly (Sunday 02:00 UTC) via pg_basebackup
- Incremental: Continuous WAL archiving to S3
- Retention: 30 days for full backups, 7 days for WAL

### 2.2 Point-in-Time Recovery (PITR)
To restore to a specific timestamp:
```sql
recovery_target_time = '2024-03-15 14:30:00 UTC'
recovery_target_action = 'promote'
```

### 2.3 Recovery Time Objective (RTO)
- Full restore: ~45 minutes for 500GB database
- PITR: ~60 minutes (depends on WAL volume)

## 3. Performance Tuning

### 3.1 Memory Configuration
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| shared_buffers | 8GB | 25% of 32GB RAM |
| effective_cache_size | 24GB | 75% of RAM |
| work_mem | 256MB | For complex sorts/joins |
| maintenance_work_mem | 2GB | For VACUUM/CREATE INDEX |

### 3.2 Connection Pooling
PgBouncer configuration:
- pool_mode = transaction
- max_client_conn = 500
- default_pool_size = 25

### 3.3 Query Optimization
- Use EXPLAIN ANALYZE for query plans
- Create partial indexes for filtered queries
- Use pg_stat_statements to identify slow queries

## 4. Monitoring

### 4.1 Key Metrics
- Connections: active, idle, waiting
- Replication lag (should be < 100ms)
- Cache hit ratio (target: > 99%)
- Transaction rate (TPS)
- Disk I/O: read/write IOPS and latency

### 4.2 Alerting Thresholds
| Metric | Warning | Critical |
|--------|---------|----------|
| Replication lag | > 1s | > 10s |
| Connection usage | > 80% | > 95% |
| Disk usage | > 75% | > 90% |
| Cache hit ratio | < 98% | < 95% |

## 5. Troubleshooting

### 5.1 Common Issues
1. Lock contention: Check pg_locks for blocking queries
2. Bloated tables: Run VACUUM FULL during maintenance window
3. Slow queries: Check missing indexes via pg_stat_user_tables
4. Connection exhaustion: Increase PgBouncer pool or fix connection leaks
""",
    }

    for filename, content in manuals.items():
        doc_path = docs_dir / filename
        with open(doc_path, "w") as f:
            f.write(content)
        print(f"  ✓ Created: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for Vectorless RAG Benchmark")
    parser.add_argument("--domain", choices=["finance", "legal", "technical", "all"], default="all")
    args = parser.parse_args()

    print("🚀 Vectorless RAG Benchmark — Dataset Downloader")
    print("=" * 55)

    if args.domain in ("all", "finance"):
        download_financebench()
    if args.domain in ("all", "legal"):
        download_cuad()
    if args.domain in ("all", "technical"):
        download_technical()

    print("\n" + "=" * 55)
    print("✅ All datasets downloaded!")
    print(f"   Location: {DATA_RAW}")
    print("\nNext steps:")
    print("  1. python -c 'from src.corpus import CorpusPreprocessor; ...'")
    print("  2. python scripts/run_benchmark.py --domain finance")


if __name__ == "__main__":
    main()
