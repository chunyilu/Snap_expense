# SnapExpense — 拍據智慧記帳與異常消費警示系統規格書與實作指南

本文件包含「SnapExpense」系統架構、資料流程、Pydantic v2 資料驗證模型、SQLite 資料庫綱要，以及可直接複製給 AI Agent / Coding Assistant 執行完整開發的 Prompt 指令。

---

## 1. 系統架構 (System Architecture)

採用分層解耦架構，將展示層、核心推論與校驗層、資料庫與外部 API 分離，確保單元測試性與容錯力：

```
+-----------------------------------------------------------------------------------+
|                           Presentation Layer (Streamlit)                          |
|  - 拍照/收據圖片上傳 (st.camera_input / st.file_uploader)                            |
|  - 辨識明細預覽與互動式修正表單 (st.data_editor / st.form)                              |
|  - 即時異常警示 Banner (st.warning / st.error)                                     |
|  - 本月預算進度條與消費類別圓餅圖 (Plotly / st.progress)                              |
+----------------------------------------+------------------------------------------+
                                         | (PIL Image / Bytes)
                                         v
+-----------------------------------------------------------------------------------+
|                         Application Core & Validation Layer                       |
|                                                                                   |
|  [ Vision Extractor ]              [ Schema Validator ]    [ Business Engine ]    |
|  - Base64 編碼轉換                 - Pydantic v2 解析     - 幣別自動換算 (TWD基準)   |
|  - 調用 GPT-4o-mini Vision         - 欄位完整度檢核       - 預算超支警示檢核         |
|  - Prompt 加入 Structured Outputs   - 金額/稅額平衡驗證    - 重複發票號碼比對         |
+-------------------+--------------------+--------------------+---------------------+
                    |                    |                    |
                    v                    v                    v
+-------------------+---+    +-----------+--------+    +------+---------------------+
|   External APIs       |    | Local SQLite DB    |    | Local Settings & Cache     |
|   - OpenAI Vision API |    | - receipts 表      |    | - .env (API Keys)          |
|   - ExchangeRate API  |    | - expense_items 表 |    | - category_budgets.json    |
+-----------------------+    +--------------------+    +----------------------------+
```

---

## 2. 資料流程 (Data Flow)

端到端處理鏈確保「非結構化影像」轉換為「具備型別安全之結構化資料」，並落實業務防呆：

```
[使用者上傳發票/收據照片 (JPG/PNG)]
           |
           v
[圖片預處理與 Base64 轉換]
  - 壓縮過大圖檔至長邊 2048px 以內
  - 轉為 Base64 格式
           |
           v
[OpenAI API (GPT-4o-mini Vision)]
  - System Prompt：要求遵循收據欄位定義
  - 注入 json_schema (Structured Outputs 保證格式不失真)
           |
           v
[Pydantic v2 資料驗證與正規化 (Parse & Validate)]
  - 驗證失敗：拋出警告並回退手動修正表單
  - 驗證成功：產出 ReceiptSchema 物件
           |
           v
[業務邏輯檢核與增強 (Business Rules Engine)]
  - 匯率轉換：若 currency != 'TWD'，調用匯率 API 換算新台幣金額
  - 異常偵測：
      * 單筆大額警示：單一項目金額 > 預設門檻 (如 3,000 TWD)
      * 類別超支警示：當月累積金額 + 本次金額 > 當月該類別預算
      * 數學一致性檢驗：SUM(items.total_price) == subtotal
           |
           v
[資料持久化與前端即時渲染 (Streamlit + SQLite)]
  - 寫入 SQLite 資料庫 (關聯儲存至 receipts 與 expense_items)
  - 刷新動態儀表板 (類別圓餅圖、預算剩餘額度、異常高亮列表)
```

---

## 3. Pydantic v2 驗證模型程式碼

```python
from datetime import date
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ExpenseCategory(str, Enum):
    MEALS = "餐飲食品"
    TRANSPORT = "交通運輸"
    ENTERTAINMENT = "休閒娛樂"
    UTILITIES = "水電與通訊"
    GROCERIES = "生活雜貨"
    MEDICAL = "醫療保健"
    OFFICE = "辦公文具"
    OTHER = "其他支出"


class ExpenseItem(BaseModel):
    name: str = Field(..., description="商品或服務名稱")
    unit_price: float = Field(..., ge=0, description="單價")
    quantity: int = Field(default=1, ge=1, description="數量")
    total_price: float = Field(..., ge=0, description="該品項小計金額 (單價 * 數量)")
    category: ExpenseCategory = Field(
        default=ExpenseCategory.OTHER, 
        description="自動推斷之消費分類"
    )

    @model_validator(mode="after")
    def verify_item_total(self):
        calculated = round(self.unit_price * self.quantity, 2)
        if abs(calculated - self.total_price) > 0.5:
            self.total_price = calculated
        return self


class ReceiptSchema(BaseModel):
    merchant_name: str = Field(..., description="店家/開立機構名稱")
    invoice_number: Optional[str] = Field(
        default=None, 
        description="發票號碼（如台灣統一發票兩碼英文+八碼數字，若無則為 None）"
    )
    transaction_date: date = Field(..., description="交易日期，格式 YYYY-MM-DD")
    currency: str = Field(default="TWD", description="幣別代碼，例如 TWD, JPY, USD")
    items: List[ExpenseItem] = Field(..., min_length=1, description="購買品項明細列表")
    tax_amount: float = Field(default=0.0, ge=0, description="稅額 (若無標示則為 0)")
    total_amount: float = Field(..., gt=0, description="收據/發票總計金額")
    confidence_score: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="模型對此張收據辨識完整度之自我評估信心分數 (0.0 ~ 1.0)"
    )

    @field_validator("invoice_number")
    @classmethod
    def clean_invoice_number(cls, v: Optional[str]) -> Optional[str]:
        if v:
            return v.strip().replace(" ", "").upper()
        return v

    @model_validator(mode="after")
    def verify_receipt_math(self):
        items_sum = sum(item.total_price for item in self.items)
        expected_total = items_sum + self.tax_amount
        if abs(expected_total - self.total_amount) > 1.0:
            # 保留容錯標記，供業務層觸發警示
            pass
        return self


class AnomalyReport(BaseModel):
    is_anomaly: bool = False
    warning_messages: List[str] = Field(default_factory=list)
```

---

## 4. SQLite 資料庫綱要 (Database Schema)

```sql
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
```

---

## 5. 給 AI Coding Agent 的實作任務 Prompt

將以下區塊整段傳送給 AI Agent（如 Claude Code, Cursor, Copilot 等），即可直接執行開發：

```markdown
# Role & Instruction
你是一位資深 Python 全端開發工程師。請依據上述架構、Pydantic 模型與 SQLite Schema，實作一個名為 `SnapExpense` 的完整可運作應用程式。

### 專案交付要求：
1. **目錄結構**：
   - `app.py`：Streamlit 前端入口（包含側邊欄配置、圖片上傳/拍照、辨識結果編輯與確認儲存、月度儀表板與圓餅圖）。
   - `extractor.py`：串接 OpenAI GPT-4o-mini Vision API，實作影像壓縮、Base64 編碼，並利用 OpenAI `client.beta.chat.completions.parse` 結合 `ReceiptSchema` 實現 Structured Outputs。
   - `database.py`：SQLite 連線與 CRUD 操作（初始化 Table、寫入收據與品項、依月份統計各類別總金額）。
   - `business_rules.py`：異常偵測邏輯（大額消費警示、類別預算超支警示、重複發票檢核、匯率換算）。
   - `requirements.txt`：列出必要套件（streamlit, openai, pydantic>=2.0, pillow, plotly, python-dotenv, requests）。
   - `.env.example`：提供環境變數設定範本（如 `OPENAI_API_KEY`、`BUDGET_MEALS` 等）。

2. **開發規範與容錯性**：
   - 全面使用 Type Hints 與清晰的 Docstrings。
   - 當未設定 `OPENAI_API_KEY` 或 API 呼叫失敗時，提供 Mock Data 按鈕或友好錯誤提示，讓展示流程不中斷。
   - 資料庫操作使用 Context Manager (`with sqlite3.connect(...)`) 確保交易安全。
   - 前端呈現須具備高度專業度：左欄顯示收據圖片與提取結果，右欄顯示警示 Banner 與即時可視化圖表。

請現在開始輸出完整且可立即運作的各檔案程式碼。
```