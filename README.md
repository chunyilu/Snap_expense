# 🧾 SnapExpense — 拍據智慧記帳與異常消費警示系統

**SnapExpense** 是一套結合 **OpenAI GPT-4o-mini Vision (Structured Outputs)**、**Pydantic v2 強型別驗證**、**SQLite 本地資料庫**與 **Streamlit 互動式儀表板**的智慧記帳與費用稽核系統。使用者只需拍照或上傳發票／收據影像，系統即可自動提取商家、日期、發票號碼、幣別與購買明細，並即時執行匯率換算與多維度異常消費偵測。

---

## ✨ 核心功能特色

- **📸 AI 視覺結構化辨識 (Vision Structured Outputs)**
  - 自動修正手機照片 EXIF 方向並將長邊最佳化壓縮至 `2048px` 以內。
  - 透過 OpenAI `client.beta.chat.completions.parse` 搭配 Pydantic `ReceiptSchema`，確保回傳格式 100% 符合資料契約。
- **🛡️ Pydantic v2 資料驗證與自動校正**
  - 自動正規化台灣統一發票號碼與幣別代碼。
  - 品項小計自動校驗（當 `單價 × 數量` 與 `小計` 誤差超過 `0.5` 時自動修正）。
  - 支援 8 大預設消費分類：`餐飲食品`、`交通運輸`、`休閒娛樂`、`水電與通訊`、`生活雜貨`、`醫療保健`、`辦公文具`、`其他支出`。
- **🚨 即時業務規則與異常消費警示引擎**
  - **外幣自動匯率換算**：非 `TWD` 收據（如 `JPY`、`USD`、`EUR` 等）自動串接公開 ExchangeRate-API 換算為新台幣，並具備離線 Fallback 匯率容錯機制。
  - **重複發票檢核**：比對歷史資料庫，防止同一張發票重複報帳或入帳。
  - **數學一致性檢驗**：檢核 `SUM(品項小計) + 稅額 == 收據總金額`，若有差額即時發出警示。
  - **單筆大額消費警示**：單一品項折合新台幣超過設定門檻（預設 `NT$ 3,000`）時主動提醒。
  - **類別月度預算超支警示**：計算「當月該類別已累積支出 + 本次收據金額」，若超過月度預算立即標示超支金額。
- **📊 動態視覺化預算儀表板**
  - 雙欄專業佈局：左欄提供影像上傳、明細表格互動式編輯 (`st.data_editor`) 與入帳確認；右欄即時渲染異常警示 Banner、本月各類別預算進度條 (`st.progress`)、Plotly 消費佔比圓餅圖與歷史收據管理。
- **🧪 內建展示與離線測試模式 (Mock Data)**
  - 未設定 `OPENAI_API_KEY` 時，可直接使用側邊欄的 4 種預設測試情境（一般餐飲收據、單筆大額警示、日本外幣收據、金額不平衡收據）完整體驗所有功能。

---

## 🏗️ 系統架構與目錄結構

```text
Snap_expense/
├── app.py                    # Streamlit 前端主程式（雙欄互動介面、表單編輯與月度儀表板）
├── models.py                 # Pydantic v2 資料模型（ExpenseCategory, ExpenseItem, ReceiptSchema, AnomalyReport）
├── extractor.py              # 影像預處理、Base64 編碼、OpenAI Vision API 串接與 Mock Data 產生器
├── business_rules.py         # 匯率換算、重複發票比對、金額平衡、大額與類別預算超支偵測引擎
├── database.py               # SQLite 連線管理 (Context Manager) 與收據/品項 CRUD 及月度統計查詢
├── category_budgets.json     # 各消費類別月度預算設定檔 (TWD)
├── requirements.txt          # Python 套件相依清單
├── .env.example              # 環境變數設定範本
└── snapexpense_architecture_and_implementation_prompt.md  # 系統規格書
```

---

## 🚀 快速開始 (Quick Start)

### 1. 建立虛擬環境並安裝套件

使用 `uv`（推薦）：

```bash
uv venv
uv pip install -r requirements.txt
```

或使用標準 `venv` + `pip`：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 設定環境變數（選用）

複製 `.env.example` 為 `.env`，並填入您的 OpenAI API Key（若僅使用內建 Mock Data 展示模式則可略過此步驟，或直接於網頁側邊欄輸入）：

```bash
cp .env.example .env
```

`.env` 設定內容範例：

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini
SINGLE_ITEM_ALERT_THRESHOLD=3000.0
DATABASE_PATH=snapexpense.db
```

### 3. 啟動應用程式

```bash
.venv/bin/streamlit run app.py
```

啟動後請開啟瀏覽器前往：`http://localhost:8501`

---

## 💡 操作流程說明

1. **上傳或載入收據**
   - **AI 辨識模式**：於左欄上傳收據照片（JPG/PNG/WEBP）或使用相機拍攝，點擊 **「🔍 使用 AI Vision 解析收據」**。
   - **離線展示模式**：於左側邊欄「🧪 展示與離線測試 (Mock Data)」點擊任一情境按鈕（如 `🧋 一般餐飲收據`、`🚄 單筆大額警示`、`💴 日本外幣收據`、`⚠️ 金額不平衡`）。
2. **檢視警示與互動修正**
   - 右欄上方會即時顯示原幣金額、對台幣匯率、折合新台幣總額，以及所有觸發的**異常消費警示 Banner**。
   - 左欄下方表單可直接修改商家名稱、日期、發票號碼，或於品項表格雙擊修改單價、數量、分類（支援動態新增與刪除列），點擊 **「🔄 更新預覽與重新檢核」** 即可即時重新驗證。
3. **確認入帳與查看儀表板**
   - 點擊 **「💾 確認無誤並寫入資料庫」** 後，收據與明細將安全寫入本地 SQLite (`snapexpense.db`)。
   - 右欄下方的**月度預算進度條**、**Plotly 消費類別圓餅圖**與**已入帳收據紀錄**將立即同步刷新。

---

## 🗄️ 資料庫綱要 (SQLite Schema)

系統啟動時會自動初始化以下兩張關聯資料表（啟用 `ON DELETE CASCADE` 外鍵約束）：

- **`receipts`（收據主表）**：儲存商家名稱、`invoice_number`、`transaction_date`、`currency`、`exchange_rate_to_twd`、`total_amount`、`total_amount_twd`、`tax_amount`、`confidence_score` 與影像路徑。
- **`expense_items`（消費品項明細表）**：透過 `receipt_id` 關聯至 `receipts`，儲存 `item_name`、`category`、`unit_price`、`quantity` 與 `total_price`。
