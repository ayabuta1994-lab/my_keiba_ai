import io
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="競馬AI 過去データ一括登録", layout="wide")

st.title("📥 過去レース結果データ 一括登録アプリ")
st.caption("AI変換プロンプト等で作成したTSV/CSVテキストを貼り付けて、スプレッドシートへ一括書き込みします。")

# 19列の標準ヘッダー定義
COLUMNS_19 = [
    "Execution Date", "Race Day", "Race Name", "Venue", "Distance", 
    "Track Condition", "Weather", "Bracket", "Horse Number", "Horse Name", 
    "Sex", "Horse Age", "Burden Weight", "Jockey", "Trainer", 
    "Tactics", "AI_Probability", "AI_Rank", "Placing"
]

# 1. テキスト入力エリア
st.subheader("1. データテキストの貼り付け")
raw_data = st.text_area(
    "AIで変換したタブ区切り（TSV）またはカンマ区切り（CSV）テキストをここに貼り付けてください：",
    height=250,
    placeholder="Execution Date\tRace Day\tRace Name...\n\t2021-07-04\tラジオNIKKEI賞..."
)

# 2. データの自動整形とプレビュー
if raw_data.strip():
    try:
        # 区切り文字の自動判定（タブ優先）
        sep = '\t' if '\t' in raw_data else ','
        
        # StringIOでテキストをPandasデータフレームに読み込み
        df = pd.read_csv(io.StringIO(raw_data), sep=sep, dtype=str)
        
        # 不足している列があれば空欄で補填
        for col in COLUMNS_19:
            if col not in df.columns:
                df[col] = ""
                
        # 19列の標準フォーマットに整える＆欠損値を空文字に置換
        df = df[COLUMNS_19].fillna("")
        
        st.subheader("2. データプレビュー（確認）")
        st.success(f"正しく解析されました： **{len(df)} 行** のデータ")
        st.dataframe(df, use_container_width=True)
        
        # 3. スプレッドシートへの追記実行ボタン
        st.subheader("3. スプレッドシートへ登録")
        if st.button("🚀 このデータをスプレッドシート（predictions）に書き込む", type="primary"):
            with st.spinner("スプレッドシートに書き込み中..."):
                try:
                    # st.secrets から Google Service Account 認証情報を取得
                    creds_dict = dict(st.secrets["gcp_service_account"])
                    scopes = [
                        "https://www.googleapis.com/auth/spreadsheets",
                        "https://www.googleapis.com/auth/drive"
                    ]
                    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
                    gc = gspread.authorize(creds)
                    
                    # スプレッドシートの取得
                    spreadsheet_id = st.secrets["spreadsheet_id"]
                    sh = gc.open_by_key(spreadsheet_id)
                    worksheet = sh.worksheet("predictions")
                    
                    # リスト形式に変換して一括書き込み
                    values = df.values.tolist()
                    worksheet.append_rows(values, value_input_option="USER_ENTERED")
                    
                    st.balloons()
                    st.success(f"🎉 正常に {len(values)} 件のデータがスプレッドシートへ追加されました！")
                    
                except Exception as e:
                    st.error(f"スプレッドシートへの書き込み時にエラーが発生しました: {e}")
                    
    except Exception as e:
        st.warning(f"データの解析に失敗しました。貼り付けた形式（タブ区切り/CSV）を確認してください: {e}")