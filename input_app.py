# -*- coding: utf-8 -*-
"""
📥 過去レース結果データ 一括登録アプリ（タブ選択対応版・血統タブ除外）
"""
import io
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="過去レースデータ一括登録", layout="wide")

st.title("📥 過去レース結果データ 一括登録アプリ")
st.caption("スプレッドシートのレース用タブを動的に取得し、書き込み先のタブを選択して一括登録できます。")

# predictions タブ用の標準カラム定義
COLUMNS_PREDICTIONS = [
    "Execution Date", "Race Name", "Venue", "Surface", "Distance", 
    "Track Condition", "Weather", "Pace", "High Pace Prob", "Front Ratio",
    "Rank", "Horse Number", "Horse Name", "Sire", "Dam's Sire", "Tactics", "AI Probability"
]

@st.cache_resource
def get_spreadsheet_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)
    spreadsheet_id = st.secrets["spreadsheet_id"]
    return gc.open_by_key(spreadsheet_id)

# スプレッドシートからタブ一覧を取得（※血統用の "horses_master" タブはレースデータ側から除外）
try:
    sh = get_spreadsheet_client()
    worksheet_list = [ws.title for ws in sh.worksheets() if ws.title != "horses_master"]
except Exception as e:
    st.error(f"スプレッドシートへの接続またはタブ一覧の取得に失敗しました: {e}")
    worksheet_list = ["predictions"]

st.subheader("1. 書き込み先タブの選択")
default_index = worksheet_list.index("predictions") if "predictions" in worksheet_list else 0
selected_worksheet = st.selectbox(
    "データを書き込むレース用タブ（ワークシート）を選択してください", 
    worksheet_list, 
    index=default_index
)

st.subheader("2. データテキストの貼り付け")
raw_data = st.text_area(
    "タブ区切り（TSV）またはカンマ区切り（CSV）テキストをここに貼り付けてください：",
    height=250,
    placeholder="Execution Date\tRace Name\tVenue..."
)

if raw_data.strip():
    try:
        sep = '\t' if '\t' in raw_data else ','
        df = pd.read_csv(io.StringIO(raw_data), sep=sep, dtype=str)
        
        for col in COLUMNS_PREDICTIONS:
            if col not in df.columns:
                df[col] = ""
                
        df = df[COLUMNS_PREDICTIONS].fillna("")
        
        st.subheader("3. データプレビュー（確認）")
        st.success(f"正しく解析されました： **{len(df)} 行** のデータ（書き込み先タブ: **{selected_worksheet}**）")
        st.dataframe(df, use_container_width=True)
        
        st.subheader("4. スプレッドシートへ登録")
        if st.button(f"🚀 「{selected_worksheet}」タブに書き込む", type="primary"):
            with st.spinner("スプレッドシートに書き込み中..."):
                try:
                    worksheet = sh.worksheet(selected_worksheet)
                    
                    values = df.values.tolist()
                    worksheet.append_rows(values, value_input_option="USER_ENTERED")
                    
                    st.balloons()
                    st.success(f"🎉 正常に {len(values)} 件のデータが 「{selected_worksheet}」 タブへ追加されました！")
                    
                except Exception as e:
                    st.error(f"スプレッドシートへの書き込み時にエラーが発生しました: {e}")
                    
    except Exception as e:
        st.warning(f"データの解析に失敗しました: {e}")
