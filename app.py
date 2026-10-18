# -*- coding: utf-8 -*-
"""
🏇 自分専用 競馬予想AI（オールインワン版・スマホ表示最適化＆21列完全統一対応）
- スマホ画面でも横スクロール不要で一目で見やすい「カード表示」UIに対応
- 全処理（予想・ログ保存・一括登録・過去参照）を21列フォーマットで統一
- CSVダウンロード機能、NaNエラー対策、ソート・検索フィルター、ペース診断、モデル前処理を完全維持
"""
import io
import pickle
import re
import traceback
import zlib
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from streamlit_gsheets import GSheetsConnection

# ページ設定（ブラウザアイコンを馬の顔 🐴 に設定）
st.set_page_config(page_title="競馬予想AI オールインワン", page_icon="🐴", layout="wide")

MODEL_PATH = Path(__file__).parent / "model.pkl"
JST = timezone(timedelta(hours=9))

# 統一する21列の固定ヘッダー定義
COLUMNS_21 = [
    "Execution Date", "Race Day", "Race Name", "Venue", "Distance",
    "Track Condition", "Weather", "Bracket", "Horse Number", "Horse Name",
    "Sex", "Horse Age", "Burden Weight", "Jockey", "Trainer",
    "Tactics", "Last 3F", "AI_Probability", "AI_Rank", "Placing", "Track_Type"
]

NUM_DEFAULTS = {
    "Bracket": 0, "Horse Number": 0, "Horse Age": 4, "Burden Weight": 55,
    "Last 3F": "mean:36.5", "Distance": 1800,
}
STR_COLS = ["Horse Name", "Sex", "Jockey", "Trainer", "Venue", "Track Condition", "Weather"]
COLUMNS_HORSES = ["Horse Name", "Sire", "Dam's Sire"]

VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉", "その他"]
TRACKS = ["良", "稍重", "重", "不良"]
WEATHERS = ["晴", "曇", "小雨", "雨", "雪"]
TRACK_TYPES = {"ダート": "1", "芝": "0", "障害": "2"}
TACTICS_LABEL = {1: "逃げ", 2: "先行", 3: "差し", 4: "追込"}


def norm(s) -> str:
    return re.sub(r"[\s_\-]", "", str(s)).lower()


@st.cache_resource(show_spinner="AIの頭脳を読み込み中…")
def load_model():
    if not MODEL_PATH.exists():
        return None, {}
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)

    meta = {"features": None, "encoders": {}}
    model = obj

    if isinstance(obj, dict):
        model = None
        for k in ("model", "pipeline", "clf", "estimator", "booster", "lgbm", "xgb"):
            if k in obj:
                model = obj[k]
                break
        if model is None:
            model = next((v for v in obj.values() if hasattr(v, "predict")), None)
        for k in ("features", "feature_names", "feature_cols", "feature_columns", "columns"):
            if k in obj and obj[k] is not None:
                meta["features"] = list(obj[k])
                break
        for k in ("encoders", "label_encoders", "encoder", "le_dict"):
            if k in obj and isinstance(obj[k], dict):
                meta["encoders"] = obj[k]
                break
    elif isinstance(obj, (tuple, list)):
        model = next((v for v in obj if hasattr(v, "predict")), None)
        for v in obj:
            if isinstance(v, dict) and not meta["encoders"]:
                meta["encoders"] = v
            elif isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
                meta["features"] = list(v)

    return model, meta


@st.cache_resource
def get_gspread_client():
    creds_dict = None
    for key in ["connections.gsheets", "gcp_service_account", "gsheets"]:
        if key in st.secrets:
            creds_dict = dict(st.secrets[key])
            break

    if creds_dict is None:
        if hasattr(st.secrets, "connections") and "gsheets" in st.secrets.connections:
            creds_dict = dict(st.secrets.connections.gsheets)
        else:
            raise KeyError("Streamlit Secrets にサービスアカウント情報が見つかりません。")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)
    
    spreadsheet_ref = creds_dict.get("spreadsheet") or st.secrets.get("spreadsheet_id")
    if not spreadsheet_ref:
        raise ValueError("Secrets に spreadsheet のURLまたはIDが設定されていません。")
    
    if "spreadsheets/d/" in str(spreadsheet_ref):
        return gc.open_by_url(spreadsheet_ref)
    else:
        return gc.open_by_key(spreadsheet_ref)


def parse_input_table(text: str) -> pd.DataFrame:
    text = text.strip().replace("```csv", "").replace("```", "").replace("\u3000", " ").strip()
    if not text:
        raise ValueError("入力データが空です。")
    first = text.splitlines()[0]
    sep = "\t" if "\t" in first else ","
    
    df = pd.read_csv(StringIO(text), sep=sep, dtype=str, skipinitialspace=True, on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]

    for col in COLUMNS_21:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNS_21].fillna("").copy()


def parse_tactics(v):
    if pd.isna(v) or v is None:
        return np.nan
    s = str(v).strip()
    m = re.match(r"^([1-4])(?:\.0+)?$", s)
    if m:
        return float(m.group(1))
    for key, code in (("逃", 1), ("先", 2), ("差", 3), ("追", 4)):
        if key in s:
            return float(code)
    return np.nan


def to_number(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(s.str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce")


def preprocess(df: pd.DataFrame, default_values: dict) -> pd.DataFrame:
    df = df.copy()
    for k, v in default_values.items():
        if k in df.columns:
            df[k] = df[k].replace("", str(v)).fillna(str(v))

    for c in NUM_DEFAULTS:
        if c in df.columns:
            df[c] = to_number(df[c])
            rule = NUM_DEFAULTS[c]
            if isinstance(rule, str):
                kind, fallback = rule.split(":")
                base = df[c].median() if kind == "median" else df[c].mean()
                fill = base if pd.notna(base) else float(fallback)
            else:
                fill = rule
            df[c] = df[c].fillna(fill).astype(float)

    code = df["Tactics"].map(parse_tactics)
    df["TacticsCode"] = code
    df["脚質ラベル"] = code.map(lambda x: TACTICS_LABEL.get(int(x), "不明") if pd.notna(x) else "不明")
    med = code.median()
    df["Tactics"] = code.fillna(med if pd.notna(med) else 2.0).astype(float)

    for c in STR_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("不明").astype(str).str.strip()

    if (df["Horse Number"] == 0).all():
        df["Horse Number"] = np.arange(1, len(df) + 1, dtype=float)

    return df


def diagnose_pace(df: pd.DataFrame) -> dict:
    code = df["TacticsCode"].dropna()
    known = len(code)
    total = len(df)
    if known == 0:
        return {"ok": False, "total": total}

    n_nige = int((code == 1).sum())
    n_senko = int((code == 2).sum())
    n_sashi = int((code == 3).sum())
    n_oikomi = int((code == 4).sum())
    front_ratio = (n_nige + n_senko) / known

    z = 10.0 * (front_ratio - 0.42) + 0.9 * (n_nige - 1.5)
    high_prob = float(np.clip(1.0 / (1.0 + np.exp(-z)), 0.01, 0.99)) * 100

    if high_prob >= 60:
        pace = "ハイペース"
    elif high_prob <= 35:
        pace = "スローペース"
    else:
        pace = "ミドルペース"

    return {
        "ok": True, "pace": pace, "high_prob": round(high_prob, 1),
        "front_ratio": round(front_ratio * 100, 1),
        "n_nige": n_nige, "n_senko": n_senko, "n_sashi": n_sashi, "n_oikomi": n_oikomi,
        "known": known, "total": total,
    }


def render_pace(p: dict):
    st.subheader("③ 展開予想・ペース診断")
    if not p.get("ok"):
        st.info("脚質（Tactics）が読み取れなかったため、ペース診断は行えませんでした。")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("想定ペース", p["pace"])
    m2.metric("ハイペース確率", f"{p['high_prob']:.1f}%")
    m3.metric("逃げ・先行の割合", f"{p['front_ratio']:.1f}%")


def build_X(df, feats):
    lookup = {norm(c): c for c in df.columns}
    X = pd.DataFrame(index=df.index)
    missing = []
    for f in feats:
        src = lookup.get(norm(f))
        if src is None:
            X[f] = 0.0
            missing.append(f)
        else:
            X[f] = df[src].values
    return X, missing


def apply_encoder(series, enc):
    if isinstance(enc, dict):
        mapping = enc
    elif hasattr(enc, "classes_"):
        mapping = {v: i for i, v in enumerate(enc.classes_)}
    else:
        return None
    return series.map(mapping).fillna(-1).astype(float)


def encode_X(X, model, meta):
    warns = []
    cat_feats = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype).startswith("string")]
    if hasattr(model, "steps"):
        return X, warns

    for c in cat_feats:
        enc = meta["encoders"].get(c) if meta["encoders"] else None
        coded = apply_encoder(X[c], enc) if enc is not None else None
        if coded is None:
            coded = X[c].map(lambda v: float(zlib.crc32(str(v).encode()) % 1000))
            warns.append(c)
        X[c] = coded
    return X, warns


def predict_top3(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X))
        if p.ndim == 2:
            classes = getattr(model, "classes_", None)
            idx = p.shape[1] - 1
            if classes is not None:
                cl = list(classes)
                if 1 in cl:
                    idx = cl.index(1)
            return p[:, idx]
        return p
    return np.clip(np.asarray(model.predict(X), dtype=float), 0, 1)


def get_feature_names_custom(model, meta):
    if meta.get("features"):
        return meta["features"]
    for attr in ("feature_names_in_", "feature_name_", "feature_names_", "feature_names"):
        v = getattr(model, attr, None)
        if v is not None and not callable(v):
            try:
                return list(v)
            except TypeError:
                pass
    return None


# --- 📱 スマホ表示用カードUIコンポーネント ---
def render_cards_prediction(df: pd.DataFrame):
    """予想結果（Tab1）をスマホで見やすいカード一覧として表示"""
    for _, row in df.iterrows():
        rank = row.get("AI_Rank", "-")
        prob = row.get("AI_Probability", "-")
        horse = row.get("Horse Name", "-")
        h_num = row.get("Horse Number", "-")
        bracket = row.get("Bracket", "-")
        jockey = row.get("Jockey", "-")
        trainer = row.get("Trainer", "-")
        sex = row.get("Sex", "")
        age = row.get("Horse Age", "")
        weight = row.get("Burden Weight", "")
        tactics = row.get("Tactics", "")
        last3f = row.get("Last 3F", "")

        # ランクに応じたアイコンバッジ
        badge = "🥇" if str(rank) == "1" else ("🥈" if str(rank) == "2" else ("🥉" if str(rank) == "3" else "🏇"))

        with st.container(border=True):
            col_left, col_right = st.columns([2, 1])
            with col_left:
                st.markdown(f"### {badge} {h_num}番 **{horse}**")
                st.caption(f"{bracket}枠 | {sex}{age} | 斤量: {weight}kg | {jockey} ({trainer})")
                st.text(f"脚質: {tactics} | 上がり3F: {last3f}s")
            with col_right:
                st.metric("AI順位", f"{rank}位")
                st.metric("勝率", f"{prob}%")


def render_cards_history(df: pd.DataFrame):
    """過去データ（Tab2）をスマホで見やすいカード一覧として表示"""
    for _, row in df.iterrows():
        r_name = row.get("Race Name", "レース")
        r_day = row.get("Race Day", "")
        venue = row.get("Venue", "")
        dist = row.get("Distance", "")
        horse = row.get("Horse Name", "")
        h_num = row.get("Horse Number", "")
        rank = row.get("AI_Rank", "-")
        prob = row.get("AI_Probability", "-")
        placing = row.get("Placing", "-")

        badge = "🥇" if str(rank) == "1" else ("🥈" if str(rank) == "2" else ("🥉" if str(rank) == "3" else "🐴"))

        with st.container(border=True):
            st.markdown(f"**{r_day} {venue} {r_name} ({dist}m)**")
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.markdown(f"{badge} **{h_num}番 {horse}**")
                st.caption(f"騎手: {row.get('Jockey', '-')} / 斤量: {row.get('Burden Weight', '-')}kg")
            with c2:
                st.metric("AI予想", f"{rank}位", f"{prob}%")
            with c3:
                st.metric("実着順", f"{placing}着" if placing else "-")


# --- アプリケーション本体 ---
st.title("🐴 自分専用 競馬予想AI（オールインワン版）")

try:
    model, meta = load_model()
    if model is None:
        st.error("model.pkl が見つからないか、有効なモデルではありません。")
        st.stop()
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 AI予想の実行", 
    "📊 過去データ参照", 
    "📥 過去レース一括登録", 
    "🧬 血統データ登録"
])

# --- TAB 1: AI予想の実行 ---
with tab1:
    st.subheader("① 共通レース条件（未入力時の自動補完用）")
    c1, c2, c3 = st.columns(3)
    with c1:
        default_race_day = st.date_input("レース開催日（Race Day）", datetime.now(JST))
        default_venue = st.selectbox("競馬場（Venue）", VENUES, index=VENUES.index("阪神"))
    with c2:
        default_track_type = st.selectbox("コース種別", ["ダート", "芝", "障害"])
        default_distance = st.number_input("距離（Distance）", value=1800, step=100)
    with c3:
        default_track_cond = st.selectbox("馬場状態（Track Condition）", TRACKS)
        default_weather = st.selectbox("天気（Weather）", WEATHERS)

    default_race_name = st.text_input("レース名（Race Name）", placeholder="例：アンタレスステークス")

    st.subheader("② 出馬表（21列フォーマット）の貼り付け")
    st.caption("※21列の固定TSV/CSVテキストを貼り付けてください。共通条件は空欄部分に自動補完されます。")
    csv_text = st.text_area("21列データテキスト", height=200, placeholder="\t".join(COLUMNS_21))

    run = st.button("🔮 AI予想を実行する", type="primary", use_container_width=True)

    if run:
        if not csv_text.strip():
            st.warning("出馬表データを貼り付けてください。")
        else:
            try:
                raw_df = parse_input_table(csv_text)
                now_str = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")

                defaults_map = {
                    "Execution Date": now_str,
                    "Race Day": str(default_race_day),
                    "Race Name": default_race_name,
                    "Venue": default_venue,
                    "Distance": str(default_distance),
                    "Track Condition": default_track_cond,
                    "Weather": default_weather,
                    "Track_Type": TRACK_TYPES.get(default_track_type, "1")
                }

                df = preprocess(raw_df, defaults_map)
                pace = diagnose_pace(df)

                feats = get_feature_names_custom(model, meta) or [
                    "Bracket", "Horse Number", "Horse Age", "Burden Weight", "Tactics", "Distance", "Last 3F"
                ]
                X, missing = build_X(df, feats)
                X, hashed = encode_X(X, model, meta)
                proba = predict_top3(model, X)
                proba_pct = np.round(np.asarray(proba, dtype=float) * 100, 1)

                df["AI_Probability"] = proba_pct.astype(str)
                rank_series = pd.Series(proba_pct).rank(ascending=False, method="min").astype(int)
                df["AI_Rank"] = rank_series.astype(str)

                df_sorted = df.sort_values("AI_Probability", ascending=False).reset_index(drop=True)
                out_21 = df_sorted[COLUMNS_21].copy().fillna("").astype(str)

                st.session_state["result_df"] = out_21
                st.session_state["pace"] = pace

                sh = get_gspread_client()
                ws = sh.worksheet("predictions")
                ws.append_rows(out_21.values.tolist(), value_input_option="USER_ENTERED")
                st.session_state["saved"] = True

            except Exception as e:
                st.session_state.pop("result_df", None)
                st.error(f"予想または保存中にエラーが発生しました: {e}")
                with st.expander("詳細ログ"):
                    st.code(traceback.format_exc())

    if "result_df" in st.session_state:
        out = st.session_state["result_df"]
        render_pace(st.session_state.get("pace", {"ok": False}))

        st.subheader(f"④ AI予想結果（全{len(out)}頭・AI_Rank順）")
        
        # 📱 表示スタイルの切替（スマホ用カード表示 / 従来テーブル表示）
        view_mode = st.radio("表示スタイル切り替え", ["📱 スマホ用カード表示", "💻 従来テーブル表示"], horizontal=True, key="pred_view_mode")
        
        if view_mode == "📱 スマホ用カード表示":
            render_cards_prediction(out)
        else:
            st.dataframe(out, hide_index=True, use_container_width=True, height=int(35 * (len(out) + 1) + 3))

        if st.session_state.get("saved"):
            st.success("✨ 予想完了＆ `predictions` タブへの21列データ蓄積が完了しました！")


# --- TAB 2: 過去データ参照（ソート・検索 ＆ スマホ表示対応） ---
with tab2:
    st.subheader("📊 過去の予想データ参照（predictions タブ）")
    if st.button("🔄 データを再読み込み"):
        st.cache_data.clear()

    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df_hist = conn.read(worksheet="predictions", ttl=0)
        if df_hist is not None and not df_hist.empty:
            df_hist = df_hist.dropna(how="all").dropna(axis=1, how="all")
            
            with st.expander("🔍 絞り込み・並び替え設定", expanded=True):
                fc1, fc2, fc3 = st.columns(3)
                with fc1:
                    filter_race_name = st.text_input("レース名で検索", placeholder="例: アンタレス")
                with fc2:
                    filter_venue = st.multiselect("競馬場で絞り込み", options=list(df_hist["Venue"].dropna().unique()) if "Venue" in df_hist.columns else [])
                with fc3:
                    sort_col = st.selectbox("並び替えの基準列", options=list(df_hist.columns), index=0)
                    ascending = st.radio("昇順/降順", options=["降順（新しい順/高ランク順）", "昇順"], index=0) == "昇順"

            df_filtered = df_hist.copy()
            if filter_race_name and "Race Name" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["Race Name"].astype(str).str.contains(filter_race_name, na=False)]
            if filter_venue and "Venue" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["Venue"].isin(filter_venue)]

            if sort_col in df_filtered.columns:
                try:
                    num_series = pd.to_numeric(df_filtered[sort_col], errors="coerce")
                    if num_series.notna().sum() > 0:
                        df_filtered["_sort_key"] = num_series
                        df_filtered = df_filtered.sort_values("_sort_key", ascending=ascending).drop(columns=["_sort_key"])
                    else:
                        df_filtered = df_filtered.sort_values(sort_col, ascending=ascending)
                except Exception:
                    df_filtered = df_filtered.sort_values(sort_col, ascending=ascending)

            st.success(f"表示中: **{len(df_filtered)} / {len(df_hist)} 件** のデータ")

            # CSVダウンロードボタン
            csv_data = df_filtered.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 表示中のデータをCSVとしてダウンロード",
                data=csv_data,
                file_name=f"predictions_export_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )

            # 📱 表示スタイルの切替
            hist_view_mode = st.radio("表示スタイル切り替え", ["📱 スマホ用カード表示", "💻 従来テーブル表示"], horizontal=True, key="hist_view_mode")
            if hist_view_mode == "📱 スマホ用カード表示":
                render_cards_history(df_filtered)
            else:
                st.dataframe(df_filtered, use_container_width=True, height=500)
        else:
            st.info("predictions タブにデータがありません。")
    except Exception as e:
        st.warning(f"読み込みエラー: {e}")


# --- TAB 3: 過去レース一括登録 ---
with tab3:
    st.subheader("📥 過去レース結果データ 一括登録（21列完全フォーマット）")
    st.caption("スプレッドシートのレース用タブ（predictions や過去別タブ）を選択して一括登録できます。（※血統タブは自動除外されます）")

    try:
        sh = get_gspread_client()
        ws_list = [ws.title for ws in sh.worksheets() if ws.title != "horses_master"]
    except Exception:
        ws_list = ["predictions"]

    def_idx = ws_list.index("predictions") if "predictions" in ws_list else 0
    target_ws = st.selectbox("書き込み先タブを選択", ws_list, index=def_idx, key="races_ws")

    raw_races = st.text_area(
        "過去レースデータの貼り付け（21列 TSV / CSV）",
        height=220,
        key="races_text",
        placeholder="\t".join(COLUMNS_21)
    )

    if raw_races.strip():
        try:
            df_r = parse_input_table(raw_races)
            st.success(f"解析成功: **{len(df_r)} 行**（書き込み先: **{target_ws}**）")
            st.dataframe(df_r, use_container_width=True)

            if st.button(f"🚀 「{target_ws}」タブに21列データを一括書き込み", type="primary"):
                with st.spinner("書き込み中..."):
                    worksheet = sh.worksheet(target_ws)
                    clean_rows = df_r.fillna("").astype(str).values.tolist()
                    worksheet.append_rows(clean_rows, value_input_option="USER_ENTERED")
                    st.balloons()
                    st.success(f"🎉 正常に {len(df_r)} 件のデータが追加されました！")
        except Exception as e:
            st.error(f"データの解析または書き込みに失敗しました: {e}")


# --- TAB 4: 血統データ登録 ＆ CSVダウンロード ---
with tab4:
    st.subheader("🧬 血統データ管理（horses_master 専用）")
    st.caption("血統データ（馬名・父・母父）の登録および学習用CSVダウンロードが行えます。")

    with st.expander("📥 登録済みの血統データをCSVダウンロード", expanded=False):
        if st.button("🔄 血統データを読み込む"):
            st.cache_data.clear()
        try:
            conn_h = st.connection("gsheets", type=GSheetsConnection)
            df_horses_hist = conn_h.read(worksheet="horses_master", ttl=0)
            if df_horses_hist is not None and not df_horses_hist.empty:
                df_horses_hist = df_horses_hist.dropna(how="all").dropna(axis=1, how="all")
                st.info(f"全 **{len(df_horses_hist)} 頭** の血統データを読み込みました。")
                
                csv_horses = df_horses_hist.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 horses_master をCSVとしてダウンロード",
                    data=csv_horses,
                    file_name=f"horses_master_export_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                st.dataframe(df_horses_hist, use_container_width=True, height=250)
            else:
                st.info("horses_master タブにデータがありません。")
        except Exception as e:
            st.warning(f"血統データの読み込みエラー: {e}")

    st.markdown("---")
    st.subheader("🧬 血統データ一括登録")
    raw_horses = st.text_area(
        "血統データの貼り付け（TSV / CSV）",
        height=220,
        key="horses_text",
        placeholder="Horse Name\tSire\tDam's Sire\nサンプルホース\tドゥラメンテ\tサンデーサイレンス"
    )

    if raw_horses.strip():
        try:
            sep = '\t' if '\t' in raw_horses else ','
            df_h = pd.read_csv(io.StringIO(raw_horses), sep=sep, dtype=str)
            for col in COLUMNS_HORSES:
                if col not in df_h.columns:
                    df_h[col] = ""
            df_h = df_h[COLUMNS_HORSES].fillna("")

            st.success(f"解析成功: **{len(df_h)} 頭** の血統データ（書き込み先: **horses_master**）")
            st.dataframe(df_h, use_container_width=True)

            if st.button("🚀 horses_master タブに書き込む", type="primary"):
                with st.spinner("書き込み中..."):
                    sh = get_gspread_client()
                    worksheet = sh.worksheet("horses_master")
                    clean_horse_rows = df_h.fillna("").astype(str).values.tolist()
                    worksheet.append_rows(clean_horse_rows, value_input_option="USER_ENTERED")
                    st.balloons()
                    st.success(f"🎉 正常に {len(df_h)} 頭の血統データが追加されました！")
        except Exception as e:
            st.error(f"データの解析または書き込みに失敗しました: {e}")
