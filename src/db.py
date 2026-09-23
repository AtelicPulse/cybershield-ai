import sqlite3
from datetime import datetime
import os

DB_PATH = 'data/audit_log.db'

def init_db(db_path=DB_PATH):
    """Initializes the SQLite database and creates the predictions table."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            flow_id TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            confidence REAL NOT NULL,
            risk_tier TEXT NOT NULL,
            action TEXT NOT NULL,
            verified INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")

def log_prediction(flow_id, prediction, confidence, risk_tier, action, db_path=DB_PATH):
    """Logs a new prediction to the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO predictions (timestamp, flow_id, prediction, confidence, risk_tier, action) VALUES (?, ?, ?, ?, ?, ?)",
        (timestamp, flow_id, prediction, confidence, risk_tier, action)
    )
    conn.commit()
    conn.close()
    # print(f"Logged prediction for flow_id: {flow_id}")

def get_recent_predictions(limit=50, db_path=DB_PATH):
    """Retrieves the most recent predictions from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()

    # Get column names for dictionary conversion
    col_names = [description[0] for description in cursor.description]

    predictions = []
    for row in rows:
        predictions.append(dict(zip(col_names, row)))
    return predictions
