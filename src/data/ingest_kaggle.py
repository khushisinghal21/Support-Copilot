"""Kaggle Dataset Ingestion: Streams real twcs.csv using DuckDB, filters AppleSupport pairs, and indexes into ChromaDB."""

import re
import json
from pathlib import Path
from typing import Optional, List, Dict
import duckdb
import typer
from rich.console import Console
from rich.table import Table

from src.config import DATA_DIR, TARGET_BRAND
from src.drafting.vector_store import HistoricalVectorStore
from src.intent.classifier import SemanticCentroidClassifier

app = typer.Typer(help="Real Kaggle twcs.csv Ingestion & ChromaDB Indexer")
console = Console()

CSV_PATH = DATA_DIR / "twcs.csv"
KAGGLE_PAIRS_JSONL = DATA_DIR / "apple_support_kaggle_pairs.jsonl"
HANDLE_STRIPPER = re.compile(r"^(@\w+\s*)+")


def clean_tweet_text(text: str) -> str:
    """Strips leading/internal Twitter user handles to obtain clean text."""
    cleaned = re.sub(r"@\w+\s*", "", text).strip()
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


@app.command()
def extract(
    csv_path: Optional[str] = typer.Option(None, "--csv", "-c", help="Path to twcs.csv"),
    max_pairs: int = typer.Option(2000, "--max-pairs", "-m", help="Maximum pairs to extract and index"),
):
    """Streams real twcs.csv via DuckDB, filters @AppleSupport pairs, and indexes into ChromaDB."""
    target_csv = Path(csv_path) if csv_path else CSV_PATH

    if not target_csv.exists():
        console.print(f"[bold red]Error: twcs.csv not found at {target_csv}[/bold red]")
        return

    console.print(f"[bold cyan]Streaming {target_csv} using DuckDB...[/bold cyan]")

    query = f"""
    SELECT 
        inbound.tweet_id AS inbound_tweet_id,
        outbound.tweet_id AS outbound_tweet_id,
        inbound.author_id AS customer_author_id,
        inbound.text AS customer_text,
        outbound.text AS agent_reply
    FROM '{target_csv}' AS inbound
    JOIN '{target_csv}' AS outbound 
      ON inbound.tweet_id = outbound.in_response_to_tweet_id
    WHERE outbound.author_id = 'AppleSupport'
      AND inbound.inbound = true
      AND inbound.in_response_to_tweet_id IS NULL
      AND length(inbound.text) > 35
      AND length(outbound.text) > 35
    LIMIT {max_pairs * 3}
    """

    con = duckdb.connect()
    rows = con.execute(query).fetchall()
    con.close()

    console.print(f"[bold green]Retrieved {len(rows)} real AppleSupport conversation pairs from Kaggle![/bold green]")

    # Filter, clean and deduplicate
    clf = SemanticCentroidClassifier()
    records: List[Dict] = []
    seen_texts = set()
    
    for row in rows:
        in_id, out_id, cust_author, raw_cust, raw_agent = row
        clean_cust = clean_tweet_text(raw_cust)
        clean_agent = clean_tweet_text(raw_agent)

        if len(clean_cust) < 25 or len(clean_agent) < 25 or clean_cust.startswith("http"):
            continue

        if clean_cust in seen_texts:
            continue
        seen_texts.add(clean_cust)

        # Classify intent for indexing metadata
        intent_res = clf.predict(clean_cust)

        records.append({
            "tweet_id": f"kaggle_{in_id}_{out_id}",
            "customer_text": clean_cust,
            "agent_reply": clean_agent,
            "intent": intent_res.primary_intent.value,
        })

        if len(records) >= max_pairs:
            break

    # Save to JSONL
    with open(KAGGLE_PAIRS_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    console.print(f"[bold green]Saved {len(records)} clean pairs to {KAGGLE_PAIRS_JSONL}[/bold green]")

    # Index into ChromaDB
    console.print(f"[bold cyan]Indexing {len(records)} real Kaggle pairs into ChromaDB vector store...[/bold cyan]")
    vs = HistoricalVectorStore()
    chunk_size = 500
    for i in range(0, len(records), chunk_size):
        chunk = records[i : i + chunk_size]
        vs.index_records(chunk)
        console.print(f"Indexed records {i} to {i + len(chunk)} / {len(records)}...")

    console.print(f"[bold green]Complete! ChromaDB now contains real historical Kaggle data.[/bold green]")


if __name__ == "__main__":
    app()
