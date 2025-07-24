import yaml
import psycopg2
from pathlib import Path

def load_config(path="config/db_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def connect_db():
    cfg = load_config()
    conn = psycopg2.connect(
        host=cfg["host"],
        port=cfg["port"],
        database=cfg["database"],
        user=cfg["user"],
        password=cfg["password"]
    )
    return conn

if __name__ == "__main__":
    try:
        conn = connect_db()
        print("Connection successful.")
        conn.close()
    except Exception as e:
        print("Connection failed:", e)
