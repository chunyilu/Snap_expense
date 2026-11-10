"""SQLite 資料庫連線與 CRUD 操作模組。

負責管理 receipts 與 expense_items 資料表初始化、收據寫入、
重複發票號碼查詢、以及依月份統計各消費類別金額（以 TWD 為基準）。
"""

import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from models import ExpenseCategory, ReceiptSchema

DEFAULT_DB_PATH = os.getenv("DATABASE_PATH", "snapexpense.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    merchant_name TEXT NOT NULL,
    invoice_number TEXT,
    transaction_date TEXT NOT NULL,
    currency TEXT NOT NULL,
    exchange_rate_to_twd REAL DEFAULT 1.0,
    total_amount REAL NOT NULL,
    total_amount_twd REAL NOT NULL,
    tax_amount REAL DEFAULT 0,
    confidence_score REAL,
    image_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS expense_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_id INTEGER NOT NULL,
    item_name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    total_price REAL NOT NULL,
    FOREIGN KEY (receipt_id) REFERENCES receipts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_receipts_date ON receipts(transaction_date);
CREATE INDEX IF NOT EXISTS idx_expense_items_cat ON expense_items(category);
"""


@contextmanager
def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """提供 SQLite 資料庫連線 Context Manager，確保外鍵約束與交易安全。

    Args:
        db_path: SQLite 資料庫檔案路徑。

    Yields:
        sqlite3.Connection: 已設定 row_factory 的資料庫連線物件。
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """初始化 SQLite 資料表與索引。

    Args:
        db_path: SQLite 資料庫檔案路徑。
    """
    with get_db_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)


def insert_receipt(
    receipt: ReceiptSchema,
    exchange_rate_to_twd: float = 1.0,
    total_amount_twd: Optional[float] = None,
    image_path: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """將驗證後的收據與品項明細寫入資料庫。

    Args:
        receipt: 通過 Pydantic 驗證的 ReceiptSchema 物件。
        exchange_rate_to_twd: 該收據幣別對新台幣之匯率。
        total_amount_twd: 換算後的新台幣總額（若未提供則自動以 total_amount * exchange_rate_to_twd 計算）。
        image_path: 選填的收據影像存檔路徑。
        db_path: SQLite 資料庫檔案路徑。

    Returns:
        int: 新增之收據記錄 ID (receipt_id)。
    """
    if total_amount_twd is None:
        total_amount_twd = round(receipt.total_amount * exchange_rate_to_twd, 2)

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO receipts (
                merchant_name,
                invoice_number,
                transaction_date,
                currency,
                exchange_rate_to_twd,
                total_amount,
                total_amount_twd,
                tax_amount,
                confidence_score,
                image_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.merchant_name,
                receipt.invoice_number,
                receipt.transaction_date.isoformat(),
                receipt.currency,
                exchange_rate_to_twd,
                receipt.total_amount,
                total_amount_twd,
                receipt.tax_amount,
                receipt.confidence_score,
                image_path,
            ),
        )
        receipt_id = int(cursor.lastrowid)

        items_payload = [
            (
                receipt_id,
                item.name,
                item.category.value if isinstance(item.category, ExpenseCategory) else str(item.category),
                item.unit_price,
                item.quantity,
                item.total_price,
            )
            for item in receipt.items
        ]
        cursor.executemany(
            """
            INSERT INTO expense_items (
                receipt_id,
                item_name,
                category,
                unit_price,
                quantity,
                total_price
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            items_payload,
        )
        return receipt_id


def check_duplicate_invoice(
    invoice_number: Optional[str],
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[Dict[str, Any]]:
    """檢查發票號碼是否已存在於資料庫中。

    Args:
        invoice_number: 欲查詢之發票號碼。
        db_path: SQLite 資料庫檔案路徑。

    Returns:
        Optional[Dict[str, Any]]: 若存在則回傳既有的收據資料字典，否則回傳 None。
    """
    if not invoice_number:
        return None
    cleaned = invoice_number.strip().replace(" ", "").upper()
    if not cleaned:
        return None

    with get_db_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, merchant_name, invoice_number, transaction_date, total_amount, currency, total_amount_twd
            FROM receipts
            WHERE UPPER(invoice_number) = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (cleaned,),
        ).fetchone()
        return dict(row) if row else None


def get_monthly_category_totals(
    year_month: str,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, float]:
    """依指定月份 (YYYY-MM) 統計各消費類別的新台幣總金額。

    Args:
        year_month: 月份字串，格式為 'YYYY-MM'（例如 '2026-11'）。
        db_path: SQLite 資料庫檔案路徑。

    Returns:
        Dict[str, float]: 類別名稱對應當月累計新台幣金額之字典（包含所有預設類別）。
    """
    totals: Dict[str, float] = {cat.value: 0.0 for cat in ExpenseCategory}

    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                ei.category AS category,
                SUM(ei.total_price * r.exchange_rate_to_twd) AS total_twd
            FROM expense_items ei
            JOIN receipts r ON ei.receipt_id = r.id
            WHERE strftime('%Y-%m', r.transaction_date) = ?
            GROUP BY ei.category
            """,
            (year_month,),
        ).fetchall()

        for row in rows:
            cat_name = row["category"]
            amount = float(row["total_twd"] or 0.0)
            totals[cat_name] = round(totals.get(cat_name, 0.0) + amount, 2)

    return totals


def get_monthly_total_twd(
    year_month: str,
    db_path: str = DEFAULT_DB_PATH,
) -> float:
    """取得指定月份 (YYYY-MM) 所有收據的新台幣總支出金額。

    Args:
        year_month: 月份字串，格式為 'YYYY-MM'。
        db_path: SQLite 資料庫檔案路徑。

    Returns:
        float: 當月累計總支出（TWD）。
    """
    with get_db_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(total_amount_twd), 0.0) AS monthly_total
            FROM receipts
            WHERE strftime('%Y-%m', transaction_date) = ?
            """,
            (year_month,),
        ).fetchone()
        return round(float(row["monthly_total"] if row else 0.0), 2)


def get_recent_receipts(
    year_month: Optional[str] = None,
    limit: int = 50,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """查詢收據歷史清單，可依月份篩選。

    Args:
        year_month: 選填的月份篩選字串 ('YYYY-MM')。
        limit: 最多回傳筆數。
        db_path: SQLite 資料庫檔案路徑。

    Returns:
        List[Dict[str, Any]]: 收據主檔與品項數量的列表。
    """
    with get_db_connection(db_path) as conn:
        if year_month:
            rows = conn.execute(
                """
                SELECT
                    r.*,
                    COUNT(ei.id) AS item_count
                FROM receipts r
                LEFT JOIN expense_items ei ON r.id = ei.receipt_id
                WHERE strftime('%Y-%m', r.transaction_date) = ?
                GROUP BY r.id
                ORDER BY r.transaction_date DESC, r.id DESC
                LIMIT ?
                """,
                (year_month, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT
                    r.*,
                    COUNT(ei.id) AS item_count
                FROM receipts r
                LEFT JOIN expense_items ei ON r.id = ei.receipt_id
                GROUP BY r.id
                ORDER BY r.transaction_date DESC, r.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def get_receipt_items(
    receipt_id: int,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """查詢單張收據的所有消費品項明細。

    Args:
        receipt_id: 收據 ID。
        db_path: SQLite 資料庫檔案路徑。

    Returns:
        List[Dict[str, Any]]: 該收據的品項列表。
    """
    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, receipt_id, item_name, category, unit_price, quantity, total_price
            FROM expense_items
            WHERE receipt_id = ?
            ORDER BY id ASC
            """,
            (receipt_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_receipt(
    receipt_id: int,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """刪除指定收據及其關聯品項（透過 ON DELETE CASCADE）。

    Args:
        receipt_id: 欲刪除之收據 ID。
        db_path: SQLite 資料庫檔案路徑。

    Returns:
        bool: 是否成功刪除記錄。
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM receipts WHERE id = ?", (receipt_id,))
        return cursor.rowcount > 0
