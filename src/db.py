"""
SQLite audit log.
"""
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/audit_log.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            flow_id TEXT,
            prediction TEXT,
            confidence REAL,
            risk_tier TEXT,
            action TEXT,
            verified INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(flow_id, prediction, confidence, risk_tier, action):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (timestamp, flow_id, prediction, confidence, risk_tier, action) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), str(flow_id), str(prediction), float(confidence), str(risk_tier), str(action))
    )
    conn.commit()
    conn.close()

def get_recent_predictions(limit=50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
