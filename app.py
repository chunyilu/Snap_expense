"""SnapExpense — 拍據智慧記帳與異常消費警示系統 Streamlit 前端入口。

左欄提供：收據拍照/圖片上傳、AI 結構化提取、互動式修正表單 (st.data_editor / st.form) 與確認入帳。
右欄提供：即時異常消費警示 Banner、本月各類別預算進度條 (st.progress)、消費類別圓餅圖 (Plotly) 與歷史帳務明細。
"""

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from pydantic import ValidationError

from business_rules import (
    evaluate_receipt_anomalies,
    load_category_budgets,
    save_category_budgets,
)
from database import (
    DEFAULT_DB_PATH,
    delete_receipt,
    get_monthly_category_totals,
    get_monthly_total_twd,
    get_receipt_items,
    get_recent_receipts,
    init_db,
    insert_receipt,
)
from extractor import (
    extract_receipt_from_image,
    get_mock_receipt,
    preprocess_and_encode_image,
)
from models import ExpenseCategory, ExpenseItem, ReceiptSchema

load_dotenv()

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

CATEGORY_OPTIONS: List[str] = [cat.value for cat in ExpenseCategory]
CURRENCY_OPTIONS: List[str] = ["TWD", "JPY", "USD", "EUR", "HKD", "CNY", "KRW", "GBP", "SGD", "AUD", "THB"]


def init_session_state() -> None:
    """初始化 Streamlit Session State 狀態變數。"""
    if "current_receipt" not in st.session_state:
        st.session_state["current_receipt"] = None
    if "current_image" not in st.session_state:
        st.session_state["current_image"] = None
    if "current_image_path" not in st.session_state:
        st.session_state["current_image_path"] = None
    if "category_budgets" not in st.session_state:
        st.session_state["category_budgets"] = load_category_budgets()
    if "single_item_threshold" not in st.session_state:
        st.session_state["single_item_threshold"] = float(
            os.getenv("SINGLE_ITEM_ALERT_THRESHOLD", "3000.0")
        )
    if "editor_version" not in st.session_state:
        st.session_state["editor_version"] = 0


def receipt_items_to_dataframe(receipt: ReceiptSchema) -> pd.DataFrame:
    """將 ReceiptSchema 的品項列表轉換為供 st.data_editor 編輯之 DataFrame。"""
    rows = []
    for item in receipt.items:
        cat_val = item.category.value if isinstance(item.category, ExpenseCategory) else str(item.category)
        rows.append(
            {
                "name": item.name,
                "category": cat_val if cat_val in CATEGORY_OPTIONS else ExpenseCategory.OTHER.value,
                "unit_price": float(item.unit_price),
                "quantity": int(item.quantity),
                "total_price": float(item.total_price),
            }
        )
    return pd.DataFrame(rows)


def build_receipt_from_form(
    merchant_name: str,
    invoice_number: Optional[str],
    transaction_date: date,
    currency: str,
    tax_amount: float,
    total_amount: float,
    confidence_score: float,
    edited_df: pd.DataFrame,
) -> ReceiptSchema:
    """從使用者編輯後的表單與 DataFrame 重建並驗證 ReceiptSchema。"""
    items: List[ExpenseItem] = []
    for _, row in edited_df.iterrows():
        name_str = str(row.get("name", "") or "").strip()
        if not name_str:
            continue
        unit_price = float(row.get("unit_price", 0.0) or 0.0)
        quantity = max(1, int(row.get("quantity", 1) or 1))
        raw_total = row.get("total_price")
        if raw_total is None or pd.isna(raw_total) or float(raw_total) == 0.0:
            total_price = round(unit_price * quantity, 2)
        else:
            total_price = float(raw_total)

        cat_str = str(row.get("category", ExpenseCategory.OTHER.value) or ExpenseCategory.OTHER.value)
        try:
            category_enum = ExpenseCategory(cat_str)
        except ValueError:
            category_enum = ExpenseCategory.OTHER

        items.append(
            ExpenseItem(
                name=name_str,
                unit_price=unit_price,
                quantity=quantity,
                total_price=total_price,
                category=category_enum,
            )
        )

    if not items:
        raise ValueError("請至少保留一筆有效的消費品項明細。")

    return ReceiptSchema(
        merchant_name=merchant_name.strip() or "未命名商家",
        invoice_number=invoice_number.strip() if invoice_number else None,
        transaction_date=transaction_date,
        currency=currency,
        items=items,
        tax_amount=float(tax_amount),
        total_amount=float(total_amount),
        confidence_score=float(confidence_score),
    )


def save_uploaded_image(img: Image.Image) -> str:
    """將處理後的收據影像儲存至本地 uploads 目錄並回傳相對路徑。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = UPLOAD_DIR / f"receipt_{timestamp}.jpg"
    img.save(file_path, format="JPEG", quality=88)
    return str(file_path)


def render_sidebar() -> Dict[str, Any]:
    """渲染側邊欄配置（API Key、異常門檻、月度預算設定、Mock Data 快速體驗）。"""
    with st.sidebar:
        st.title("⚙️ SnapExpense 設定")
        st.caption("拍據智慧記帳與異常消費警示系統")

        env_key = os.getenv("OPENAI_API_KEY", "")
        default_key = "" if env_key == "your_openai_api_key_here" else env_key

        api_key = st.text_input(
            "🔑 OpenAI API Key",
            value=default_key,
            type="password",
            help="用於呼叫 GPT-4o-mini Vision 結構化解析收據。若未設定，可使用下方「範例資料」按鈕體驗完整功能。",
        )
        model_name = st.selectbox(
            "🤖 Vision 模型",
            options=["gpt-4o-mini", "gpt-4o"],
            index=0,
        )

        st.divider()
        st.subheader("🧪 展示與離線測試 (Mock Data)")
        st.caption("無須 API Key 即可模擬各類收據辨識與異常偵測情境：")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if st.button("🧋 一般餐飲收據", width="stretch"):
                st.session_state["current_receipt"] = get_mock_receipt("standard")
                st.session_state["current_image"] = None
                st.session_state["current_image_path"] = None
                st.session_state["editor_version"] += 1
                st.rerun()
            if st.button("💴 日本外幣收據", width="stretch"):
                st.session_state["current_receipt"] = get_mock_receipt("foreign_jpy")
                st.session_state["current_image"] = None
                st.session_state["current_image_path"] = None
                st.session_state["editor_version"] += 1
                st.rerun()
        with col_m2:
            if st.button("🚄 單筆大額警示", width="stretch"):
                st.session_state["current_receipt"] = get_mock_receipt("high_value")
                st.session_state["current_image"] = None
                st.session_state["current_image_path"] = None
                st.session_state["editor_version"] += 1
                st.rerun()
            if st.button("⚠️ 金額不平衡", width="stretch"):
                st.session_state["current_receipt"] = get_mock_receipt("math_mismatch")
                st.session_state["current_image"] = None
                st.session_state["current_image_path"] = None
                st.session_state["editor_version"] += 1
                st.rerun()

        st.divider()
        st.subheader("🛡️ 異常警示與預算門檻")
        threshold = st.number_input(
            "單一品項大額警示門檻 (TWD)",
            min_value=100.0,
            max_value=500000.0,
            value=float(st.session_state["single_item_threshold"]),
            step=500.0,
        )
        st.session_state["single_item_threshold"] = threshold

        current_ym = date.today().strftime("%Y-%m")
        selected_ym = st.text_input(
            "📅 儀表板統計月份 (YYYY-MM)",
            value=current_ym,
            help="調整欲檢視的月度預算與消費統計月份，格式 YYYY-MM",
        )

        with st.expander("💰 各類別月度預算設定 (TWD)", expanded=False):
            updated_budgets: Dict[str, float] = {}
            for cat in ExpenseCategory:
                cat_name = cat.value
                current_val = float(st.session_state["category_budgets"].get(cat_name, 5000.0))
                updated_budgets[cat_name] = st.number_input(
                    f"{cat_name}",
                    min_value=0.0,
                    max_value=1000000.0,
                    value=current_val,
                    step=500.0,
                    key=f"budget_input_{cat_name}",
                )
            if st.button("💾 儲存預算設定", width="stretch"):
                st.session_state["category_budgets"] = updated_budgets
                save_category_budgets(updated_budgets)
                st.success("已更新並儲存各類別月度預算！")

        return {
            "api_key": api_key,
            "model_name": model_name,
            "single_item_threshold": threshold,
            "selected_ym": selected_ym.strip() or current_ym,
            "budgets": st.session_state["category_budgets"],
        }


def render_left_column(config: Dict[str, Any]) -> None:
    """渲染左欄：影像上傳/拍照、AI 辨識觸發、以及互動式明細修正與入帳表單。"""
    st.subheader("📸 1. 收據影像拍攝與上傳")

    tab_upload, tab_camera = st.tabs(["📁 上傳收據圖片 (JPG/PNG)", "📷 相機即時拍攝"])

    uploaded_Source = None
    with tab_upload:
        file_obj = st.file_uploader(
            "請選擇發票或收據影像檔",
            type=["jpg", "jpeg", "png", "webp"],
            key="receipt_file_uploader",
        )
        if file_obj is not None:
            uploaded_Source = file_obj

    with tab_camera:
        cam_obj = st.camera_input("對準發票或收據拍攝", key="receipt_camera_input")
        if cam_obj is not None:
            uploaded_Source = cam_obj

    if uploaded_Source is not None:
        try:
            _, processed_img = preprocess_and_encode_image(uploaded_Source.getvalue())
            st.image(processed_img, caption="收據影像預覽（已自動最佳化尺寸）", width="stretch")

            col_btn1, col_btn2 = st.columns([3, 2])
            with col_btn1:
                extract_clicked = st.button(
                    "🔍 使用 AI Vision 解析收據",
                    type="primary",
                    width="stretch",
                )
            with col_btn2:
                mock_fallback_clicked = st.button(
                    "✨ 載入範例辨識結果",
                    width="stretch",
                )

            if extract_clicked:
                with st.spinner("正在透過 GPT-4o-mini Vision 進行結構化提取與 Pydantic 驗證..."):
                    try:
                        parsed_receipt = extract_receipt_from_image(
                            image_source=uploaded_Source.getvalue(),
                            api_key=config["api_key"],
                            model=config["model_name"],
                        )
                        saved_path = save_uploaded_image(processed_img)
                        st.session_state["current_receipt"] = parsed_receipt
                        st.session_state["current_image"] = processed_img
                        st.session_state["current_image_path"] = saved_path
                        st.session_state["editor_version"] += 1
                        st.success("🎉 收據辨識成功！請於下方表單核對或微調明細。")
                    except Exception as exc:
                        st.error(f"❌ 辨識失敗：{exc}")
                        st.info("💡 提示：若您尚未配置 OpenAI API Key，可點擊旁邊的「載入範例辨識結果」或使用左側邊欄的 Mock Data 按鈕繼續體驗。")

            if mock_fallback_clicked:
                saved_path = save_uploaded_image(processed_img)
                st.session_state["current_receipt"] = get_mock_receipt("standard")
                st.session_state["current_image"] = processed_img
                st.session_state["current_image_path"] = saved_path
                st.session_state["editor_version"] += 1
                st.rerun()

        except Exception as img_exc:
            st.error(f"影像讀取失敗：{img_exc}")

    st.divider()
    st.subheader("📝 2. 辨識明細預覽與互動式修正表單")

    receipt: Optional[ReceiptSchema] = st.session_state.get("current_receipt")
    if receipt is None:
        st.info(
            "👆 請先上傳收據照片並點擊「使用 AI Vision 解析收據」，或從左側邊欄點擊「展示與離線測試 (Mock Data)」載入範例收據。"
        )
        return

    # 顯示 AI 辨識信心度指標
    conf_pct = int(receipt.confidence_score * 100)
    conf_color = "🟢" if conf_pct >= 85 else ("🟡" if conf_pct >= 70 else "🔴")
    st.caption(f"{conf_color} **AI 辨識信心分數：{conf_pct}%** — 您可直接於下方表格雙擊修改品項、分類或金額，系統將即時重新檢核。")

    editor_key = f"items_editor_{st.session_state['editor_version']}"

    with st.form("receipt_verification_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            merchant_name = st.text_input("🏪 商家名稱", value=receipt.merchant_name)
            transaction_date = st.date_input("📅 交易日期", value=receipt.transaction_date)
            tax_amount = st.number_input(
                "🧾 稅額",
                min_value=0.0,
                value=float(receipt.tax_amount),
                step=1.0,
            )
        with c2:
            invoice_number = st.text_input(
                "🔢 發票號碼 (選填)",
                value=receipt.invoice_number or "",
                placeholder="例如 AB-12345678",
            )
            curr_idx = (
                CURRENCY_OPTIONS.index(receipt.currency)
                if receipt.currency in CURRENCY_OPTIONS
                else 0
            )
            currency = st.selectbox("💱 交易幣別", options=CURRENCY_OPTIONS, index=curr_idx)
            total_amount = st.number_input(
                "💵 收據總計金額 (原幣)",
                min_value=0.01,
                value=float(receipt.total_amount),
                step=10.0,
            )

        st.markdown("##### 🛒 購買品項明細（可新增、刪除或直接修改欄位）")
        items_df = receipt_items_to_dataframe(receipt)
        edited_df = st.data_editor(
            items_df,
            key=editor_key,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "name": st.column_config.TextColumn("品項名稱", required=True),
                "category": st.column_config.SelectboxColumn(
                    "消費分類",
                    options=CATEGORY_OPTIONS,
                    required=True,
                ),
                "unit_price": st.column_config.NumberColumn(
                    "單價",
                    min_value=0.0,
                    format="%.2f",
                    required=True,
                ),
                "quantity": st.column_config.NumberColumn(
                    "數量",
                    min_value=1,
                    step=1,
                    format="%d",
                    required=True,
                ),
                "total_price": st.column_config.NumberColumn(
                    "小計 (自動依單價x數量校正)",
                    min_value=0.0,
                    format="%.2f",
                    required=True,
                ),
            },
        )

        btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 1])
        with btn_col1:
            recheck_btn = st.form_submit_button("🔄 更新預覽與重新檢核", width="stretch")
        with btn_col2:
            save_btn = st.form_submit_button(
                "💾 確認無誤並寫入資料庫",
                type="primary",
                width="stretch",
            )
        with btn_col3:
            clear_btn = st.form_submit_button("🗑️ 清空", width="stretch")

    if clear_btn:
        st.session_state["current_receipt"] = None
        st.session_state["current_image"] = None
        st.session_state["current_image_path"] = None
        st.session_state["editor_version"] += 1
        st.rerun()

    if recheck_btn or save_btn:
        try:
            updated_receipt = build_receipt_from_form(
                merchant_name=merchant_name,
                invoice_number=invoice_number,
                transaction_date=transaction_date,
                currency=currency,
                tax_amount=tax_amount,
                total_amount=total_amount,
                confidence_score=receipt.confidence_score,
                edited_df=edited_df,
            )
            st.session_state["current_receipt"] = updated_receipt

            if save_btn:
                report = evaluate_receipt_anomalies(
                    receipt=updated_receipt,
                    category_budgets=config["budgets"],
                    single_item_threshold_twd=config["single_item_threshold"],
                )
                receipt_id = insert_receipt(
                    receipt=updated_receipt,
                    exchange_rate_to_twd=report.exchange_rate_to_twd,
                    total_amount_twd=report.total_amount_twd,
                    image_path=st.session_state.get("current_image_path"),
                )
                st.toast(f"✅ 收據 #{receipt_id} ({updated_receipt.merchant_name}) 已成功存入資料庫！", icon="🎉")
                st.success(
                    f"✅ 已成功儲存收據 **#{receipt_id} {updated_receipt.merchant_name}** "
                    f"（折合 **NT$ {report.total_amount_twd:,.0f}**）！右側儀表板已同步刷新。"
                )
                st.session_state["current_receipt"] = None
                st.session_state["current_image"] = None
                st.session_state["current_image_path"] = None
                st.session_state["editor_version"] += 1
                st.rerun()
            else:
                st.toast("🔄 明細與異常檢核結果已更新！", icon="✨")
                st.rerun()

        except (ValidationError, ValueError) as val_err:
            st.error(f"⚠️ 資料驗證未通過，請修正表單欄位：{val_err}")


def render_right_column(config: Dict[str, Any]) -> None:
    """渲染右欄：即時異常警示 Banner、本月預算進度條、消費類別圓餅圖與歷史帳務清單。"""
    st.subheader("🚨 3. 即時業務規則與異常消費警示")

    receipt: Optional[ReceiptSchema] = st.session_state.get("current_receipt")
    if receipt is not None:
        report = evaluate_receipt_anomalies(
            receipt=receipt,
            category_budgets=config["budgets"],
            single_item_threshold_twd=config["single_item_threshold"],
        )

        # 顯示匯率與換算卡片
        m1, m2, m3 = st.columns(3)
        m1.metric("原幣總金額", f"{receipt.total_amount:,.2f} {receipt.currency}")
        m2.metric("換算匯率 (對 TWD)", f"{report.exchange_rate_to_twd:,.4f}")
        m3.metric("折合新台幣總額", f"NT$ {report.total_amount_twd:,.0f}")

        if report.is_anomaly:
            st.error(f"⚠️ **偵測到 {len(report.warning_messages)} 項潛在異常或預算警示，請於入帳前確認：**")
            for msg in report.warning_messages:
                if msg.startswith("🚨"):
                    st.error(msg)
                elif msg.startswith("ℹ️"):
                    st.info(msg)
                else:
                    st.warning(msg)
        else:
            for msg in report.warning_messages:
                st.info(msg)
            st.success(
                "✅ **通過所有檢核！** 此收據無重複發票、品項金額平衡、無單筆超額且未超過當月類別預算。"
            )
    else:
        st.info("目前無待審核收據。當左側載入或辨識收據時，此處將即時顯示匯率換算與異常偵測結果。")

    st.divider()
    selected_ym = config["selected_ym"]
    st.subheader(f"📊 4. {selected_ym} 月度預算與消費儀表板")

    budgets: Dict[str, float] = config["budgets"]
    monthly_cat_totals = get_monthly_category_totals(selected_ym)
    monthly_total_twd = get_monthly_total_twd(selected_ym)
    total_budget_twd = sum(budgets.values())
    remaining_budget_twd = total_budget_twd - monthly_total_twd

    # 總體月度指標
    k1, k2, k3 = st.columns(3)
    k1.metric("本月累計總支出", f"NT$ {monthly_total_twd:,.0f}")
    k2.metric("本月總預算額度", f"NT$ {total_budget_twd:,.0f}")
    k3.metric(
        "本月預算剩餘額度",
        f"NT$ {remaining_budget_twd:,.0f}",
        delta=f"{remaining_budget_twd:,.0f} TWD",
        delta_color="normal" if remaining_budget_twd >= 0 else "inverse",
    )

    overall_ratio = min(1.0, monthly_total_twd / total_budget_twd) if total_budget_twd > 0 else 0.0
    overall_pct = (monthly_total_twd / total_budget_twd * 100) if total_budget_twd > 0 else 0.0
    st.progress(
        overall_ratio,
        text=f"本月總預算使用率：{overall_pct:.1f}% (NT$ {monthly_total_twd:,.0f} / NT$ {total_budget_twd:,.0f})",
    )

    # 視覺化圖表與各類別預算進度條
    chart_tab, budget_tab, history_tab = st.tabs(
        ["🥧 消費類別佔比圓餅圖", "📏 各類別預算進度條", "🗂️ 已入帳收據紀錄"]
    )

    with chart_tab:
        active_data = [
            {"類別": cat, "金額 (TWD)": amt}
            for cat, amt in monthly_cat_totals.items()
            if amt > 0
        ]
        if active_data:
            df_chart = pd.DataFrame(active_data)
            fig = px.pie(
                df_chart,
                names="類別",
                values="金額 (TWD)",
                hole=0.42,
                title=f"{selected_ym} 各類別消費分佈 (TWD)",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>金額: NT$ %{value:,.0f}<br>佔比: %{percent}<extra></extra>",
            )
            fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=360)
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption(f"📭 `{selected_ym}` 尚無已入帳之消費資料。請於左側確認並儲存收據後即可查看動態圓餅圖。")

    with budget_tab:
        for cat in ExpenseCategory:
            cat_name = cat.value
            spent = float(monthly_cat_totals.get(cat_name, 0.0))
            limit = float(budgets.get(cat_name, 0.0))
            ratio = min(1.0, spent / limit) if limit > 0 else 0.0
            pct = (spent / limit * 100) if limit > 0 else 0.0

            status_icon = "🔴 超支" if (limit > 0 and spent > limit) else ("🟡 接近上限" if pct >= 80 else "🟢 正常")
            st.progress(
                ratio,
                text=f"{status_icon} **{cat_name}**：NT$ {spent:,.0f} / NT$ {limit:,.0f} ({pct:.1f}%)",
            )

    with history_tab:
        recent_list = get_recent_receipts(year_month=selected_ym, limit=30)
        if not recent_list:
            st.caption(f"`{selected_ym}` 尚無收據紀錄。")
        else:
            for rec in recent_list:
                rec_id = rec["id"]
                inv_str = f" | 發票: `{rec['invoice_number']}`" if rec.get("invoice_number") else ""
                header_text = (
                    f"#{rec_id} 【{rec['transaction_date']}】 **{rec['merchant_name']}** — "
                    f"NT$ {rec['total_amount_twd']:,.0f} "
                    f"({rec['total_amount']:,.2f} {rec['currency']}){inv_str}"
                )
                with st.expander(header_text, expanded=False):
                    items = get_receipt_items(rec_id)
                    if items:
                        df_items = pd.DataFrame(items)[
                            ["item_name", "category", "unit_price", "quantity", "total_price"]
                        ].rename(
                            columns={
                                "item_name": "品項名稱",
                                "category": "消費分類",
                                "unit_price": "單價",
                                "quantity": "數量",
                                "total_price": "小計",
                            }
                        )
                        st.dataframe(df_items, width="stretch", hide_index=True)
                    if st.button("🗑️ 刪除此筆收據", key=f"del_rec_{rec_id}"):
                        delete_receipt(rec_id)
                        st.toast(f"已刪除收據 #{rec_id}")
                        st.rerun()


def main() -> None:
    """Streamlit 應用程式主入口。"""
    st.set_page_config(
        page_title="SnapExpense — 拍據智慧記帳與異常消費警示系統",
        page_icon="🧾",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_db(DEFAULT_DB_PATH)
    init_session_state()

    st.title("🧾 SnapExpense — 拍據智慧記帳與異常消費警示系統")
    st.markdown(
        "結合 **GPT-4o-mini Vision Structured Outputs**、**Pydantic v2 型別驗證** 與 **即時業務異常偵測引擎**，將收據照片秒速化為結構化帳本。"
    )

    config = render_sidebar()

    left_col, right_col = st.columns([1.08, 1.0], gap="large")
    with left_col:
        render_left_column(config)
    with right_col:
        render_right_column(config)


if __name__ == "__main__":
    main()
