# -*- coding: utf-8 -*-
"""
🏇 自分専用 競馬予想AI（オールインワン版・スプレッドシート連動シンプル予想対応）
- 共通条件を選び、スプレッドシートの対象レースタブを選択してワンクリックで予想を実行
"""
import base64
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

    if isinstance(obj, dict) and isinstance(obj.get("models"), dict):
        if not all(k in obj["models"] for k in ("win", "top3")):
            raise ValueError("model.pkl の models に win / top3 がありません。")
        return obj, {"bundle": obj, "features": list(obj.get("features", [])), "encoders": obj.get("encoders", {})}

    meta = {"features": None, "encoders": {}}
    model = obj
    if isinstance(obj, dict):
        model = next((obj[k] for k in ("model", "pipeline", "clf", "estimator", "booster", "lgbm", "xgb") if k in obj), None)
        if model is None:
            model = next((v for v in obj.values() if hasattr(v, "predict")), None)
        for k in ("features", "feature_names", "feature_cols", "feature_columns", "columns"):
            if k in obj and obj[k] is not None:
                meta["features"] = list(obj[k]); break
        for k in ("encoders", "label_encoders", "encoder", "le_dict"):
            if k in obj and isinstance(obj[k], dict):
                meta["encoders"] = obj[k]; break
    elif isinstance(obj, (tuple, list)):
        model = next((v for v in obj if hasattr(v, "predict")), None)
        for v in obj:
            if isinstance(v, dict) and not meta["encoders"]: meta["encoders"] = v
            elif isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v): meta["features"] = list(v)
    return model, meta


def _norm_horse(v):
    return re.sub(r"[\s\u200b-\u200f\u2060\ufeff]", "", str(v or ""))


def _read_worksheet_dataframe(ws):
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame()
    headers = [str(v).strip() for v in values[0]]
    if not any(headers):
        return pd.DataFrame()
    while headers and not headers[-1]:
        headers.pop()
    if not headers:
        return pd.DataFrame()
    rows = []
    for row in values[1:]:
        row = list(row[:len(headers)])
        if len(row) < len(headers):
            row.extend([""] * (len(headers) - len(row)))
        rows.append(row)
    return pd.DataFrame(rows, columns=headers)


def _load_spreadsheet_history(spreadsheet):
    frames = []
    for ws in spreadsheet.worksheets():
        if ws.title == "horses_master":
            continue
        try:
            part = _read_worksheet_dataframe(ws)
        except Exception as exc:
            raise RuntimeError(f"スプレッドシートの「{ws.title}」タブを読み込めません: {exc}") from exc
        if part.empty or "Horse Name" not in part.columns:
            continue
        if not ("Race Day" in part.columns or "Placing" in part.columns):
            continue
        frames.append(ensure_21_columns(part))
    if not frames:
        return pd.DataFrame(columns=COLUMNS_21)
    return pd.concat(frames, ignore_index=True)


def _new_model_matrix(df, bundle):
    features = list(bundle["features"])
    numeric = set(bundle.get("numeric_features", []))
    categorical = set(bundle.get("categorical_features", []))
    encoders = bundle.get("encoders", {})
    spreadsheet = get_gspread_client()
    hist = _load_spreadsheet_history(spreadsheet)
    hist = ensure_21_columns(hist).drop_duplicates(
        subset=["Race Day", "Race Name", "Venue", "Horse Number", "Horse Name"], keep="last"
    )
    hist["_day"] = pd.to_datetime(hist["Race Day"], errors="coerce")
    hist["_placing"] = to_number(hist["Placing"])
    hist["_last3f"] = to_number(hist["Last 3F"])
    hist["_tactics"] = hist["Tactics"].map(parse_tactics)

    try:
        horses_ws = spreadsheet.worksheet("horses_master")
    except Exception as exc:
        raise RuntimeError("Googleスプレッドシートに「horses_master」タブがありません。") from exc
    shorses = _read_worksheet_dataframe(horses_ws)
    pedigree = {}
    if not shorses.empty and "Horse Name" in shorses.columns:
        for _, hr in shorses.fillna("").iterrows():
            pedigree[_norm_horse(hr.get("Horse Name", ""))] = (hr.get("Sire", ""), hr.get("Dam's Sire", ""))

    rows = []
    for _, r in df.iterrows():
        horse_key = _norm_horse(r.get("Horse Name", ""))
        race_day = pd.to_datetime(r.get("Race Day", ""), errors="coerce")
        past = hist[(hist["Horse Name"].map(_norm_horse) == horse_key) & hist["_day"].notna() & (hist["_day"] < race_day)]
        past = past.sort_values("_day")
        last3 = past.tail(3)
        placings = past["_placing"].dropna()
        last3f = last3["_last3f"].dropna()
        prev3f = past["_last3f"].dropna()
        sire, damsire = pedigree.get(horse_key, ("Unknown", "Unknown"))
        row = {
            "Distance": r.get("Distance", 1800), "Bracket": r.get("Bracket", 0),
            "Horse Number": r.get("Horse Number", 0), "Horse Age": r.get("Horse Age", 4),
            "Burden Weight": r.get("Burden Weight", 55), "Track_Type": r.get("Track_Type", 1),
            "Hist_Starts": float(len(placings)),
            "Hist_Win_Rate": float((placings == 1).mean()) if len(placings) else 0.0,
            "Hist_Top3_Rate": float((placings <= 3).mean()) if len(placings) else 0.0,
            "Hist_Mean_Placing3": float(last3["_placing"].dropna().mean()) if last3["_placing"].notna().any() else 0.0,
            "Hist_Last3F_Mean3": float(last3f.mean()) if len(last3f) else 0.0,
            "Hist_Last3F_Last": float(prev3f.iloc[-1]) if len(prev3f) else 0.0,
            "Hist_Last_Tactics": float(past["_tactics"].dropna().iloc[-1]) if past["_tactics"].notna().any() else 0.0,
            "Race Name": r.get("Race Name", "Unknown"), "Venue": r.get("Venue", "Unknown"),
            "Track Condition": r.get("Track Condition", "Unknown"), "Weather": r.get("Weather", "Unknown"),
            "Horse Name": r.get("Horse Name", "Unknown"), "Sex": r.get("Sex", "Unknown"),
            "Jockey": r.get("Jockey", "Unknown"), "Trainer": r.get("Trainer", "Unknown"),
            "Sire": sire or "Unknown", "Dam's Sire": damsire or "Unknown",
        }
        rows.append(row)
    X = pd.DataFrame(rows, columns=features)
    for c in features:
        if c in categorical:
            mapping = encoders.get(c, {})
            X[c] = X[c].fillna("Unknown").astype(str).map(mapping).fillna(-1).astype("int32")
        elif c in numeric:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(float)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X


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


def ensure_21_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in COLUMNS_21:
        if col not in df.columns:
            if col == "Distance":
                df[col] = "1800"
            elif col == "Burden Weight":
                df[col] = "55.0"
            elif col == "Track_Type":
                df[col] = "1"
            elif col == "AI_Probability":
                df[col] = "0.0"
            elif col == "AI_Rank":
                df[col] = "1"
            else:
                df[col] = ""
    return df[COLUMNS_21].fillna("").astype(str).copy()


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
    numeric_code = pd.to_numeric(code, errors="coerce")
    med = numeric_code.median()
    df["Tactics"] = numeric_code.fillna(med if pd.notna(med) else 2.0).astype(float)

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
    X_val = X.values if hasattr(X, "values") else X
    try:
        preds = model.predict(X_val, predict_disable_shape_check=True)
    except TypeError:
        preds = model.predict(X_val)
    return np.clip(np.asarray(preds, dtype=float), 0, 1)


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


def create_safe_download_link(df: pd.DataFrame, filename: str, label: str) -> str:
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    b64 = base64.b64encode(csv_bytes).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" target="_blank" style="display: inline-block; padding: 0.5em 1em; color: white; background-color: #FF4B4B; border-radius: 8px; text-decoration: none; font-weight: bold; margin-bottom: 10px;">{label}</a>'


def render_cards_prediction(df: pd.DataFrame):
    for _, row in df.reset_index(drop=True).iterrows():
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
                if "複勝確率(3着以内)" in row.index:
                    st.metric("複勝確率（3着以内）", f"{row.get('複勝確率(3着以内)', '-')}%")


def render_cards_history(df: pd.DataFrame):
    df_render = df.reset_index(drop=True).head(50)
    if len(df) > 50:
        st.info(f"💡 データが全 {len(df)} 件あるため、直近・上位の 50 件を表示しています。")

    for _, row in df_render.iterrows():
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

# --- TAB 1: AI予想の実行（スプレッドシート連動型） ---
with tab1:
    st.subheader("① 予想対象レースの選択（Googleスプレッドシート連動）")
    st.caption("※「Tab 3（過去レース一括登録）」等であらかじめスプレッドシートに登録した対象レースのタブを選択してください。")

    try:
        sh = get_gspread_client()
        ws_list = [ws.title for ws in sh.worksheets() if ws.title != "horses_master"]
    except Exception:
        ws_list = ["predictions"]

    target_eval_ws = st.selectbox("予想するレースのタブを選択", ws_list, key="eval_ws_select")

    run = st.button("🔮 選択したタブのデータでAI予想を実行する", type="primary", use_container_width=True)

    if run:
        try:
            conn_eval = st.connection("gsheets", type=GSheetsConnection)
            df_target_raw = conn_eval.read(worksheet=target_eval_ws, ttl=0)
            
            if df_target_raw is None or df_target_raw.empty:
                st.warning(f"「{target_eval_ws}」タブにデータがありません。")
            else:
                raw_df = ensure_21_columns(df_target_raw)
                st.session_state.pop("top3_probabilities", None)
                now_str = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")

                defaults_map = {
                    "Execution Date": now_str,
                }

                df = preprocess(raw_df, defaults_map)
                pace = diagnose_pace(df)

                top3_pct = None
                if isinstance(model, dict) and isinstance(model.get("models"), dict):
                    X = _new_model_matrix(df, model)
                    win_model = model["models"]["win"]
                    top3_model = model["models"]["top3"]
                    proba = np.asarray(win_model.predict(X, num_iteration=getattr(win_model, "best_iteration", None)), dtype=float)
                    top3_proba = np.asarray(top3_model.predict(X, num_iteration=getattr(top3_model, "best_iteration", None)), dtype=float)
                    if len(proba) != len(df) or len(top3_proba) != len(df):
                        raise ValueError("モデルの予測件数と出馬表の頭数が一致しません。")
                    top3_pct = np.round(np.clip(top3_proba, 0, 1) * 100, 1)
                else:
                    feats = get_feature_names_custom(model, meta) or [
                        "Bracket", "Horse Number", "Horse Age", "Burden Weight", "Tactics", "Distance", "Last 3F"
                    ]
                    X, missing = build_X(df, feats)
                    X, hashed = encode_X(X, model, meta)
                    proba = predict_top3(model, X)
                
                proba_pct = np.round(np.asarray(proba, dtype=float) * 100, 1)

                df["AI_Probability"] = proba_pct.astype(str)
                rank_series = pd.Series(proba_pct, index=df.index).rank(ascending=False, method="min").astype(int)
                df["AI_Rank"] = rank_series.astype(str)
                if top3_pct is not None:
                    df["複勝確率(3着以内)"] = top3_pct.astype(str)

                df_sorted = df.sort_values("AI_Probability", key=lambda col: pd.to_numeric(col, errors="coerce"), ascending=False).reset_index(drop=True)
                out_21 = ensure_21_columns(df_sorted)
                display_df = df_sorted.copy()

                st.session_state["result_df"] = out_21
                st.session_state["result_view_df"] = display_df
                st.session_state["pace"] = pace

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
        view_out = st.session_state.get("result_view_df", out)
        render_pace(st.session_state.get("pace", {"ok": False}))

        st.subheader(f"② AI予想結果（全{len(out)}頭・AI_Rank順）")
        if "複勝確率(3着以内)" in view_out.columns:
            st.caption("勝率は単勝モデル、複勝確率は3着以内モデルの予測値です。")
        
        view_mode = st.radio("表示スタイル切り替え", ["📱 スマホ用カード表示", "💻 従来テーブル表示"], horizontal=True, key="pred_view_mode")
        
        if view_mode == "📱 スマホ用カード表示":
            render_cards_prediction(view_out)
        else:
            st.dataframe(view_out, hide_index=True, use_container_width=True, height=int(35 * (len(view_out) + 1) + 3))

        if st.session_state.get("saved"):
            st.success("✨ 予想完了＆ `predictions` タブへの21列データ蓄積が完了しました！")


# --- TAB 2: 過去データ参照 ---
with tab2:
    st.subheader("📊 過去の予想データ参照（predictions タブ）")
    if st.button("🔄 データを再読み込み"):
        st.cache_data.clear()

    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df_hist_raw = conn.read(worksheet="predictions", ttl=0)
        if df_hist_raw is not None and not df_hist_raw.empty:
            df_hist_raw = df_hist_raw.dropna(how="all").dropna(axis=1, how="all")
            df_hist = ensure_21_columns(df_hist_raw)
            
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

            df_filtered = df_filtered.reset_index(drop=True)
            st.success(f"表示中: **{len(df_filtered)} / {len(df_hist)} 件** のデータ")

            filename_pred = f"predictions_export_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}.csv"
            link_html_pred = create_safe_download_link(df_filtered, filename_pred, "📥 表示中のデータをCSV保存（スマホ対応）")
            st.markdown(link_html_pred, unsafe_allow_html=True)

            hist_view_mode = st.radio("表示スタイル切り替え", ["📱 スマホ用カード表示", "💻 従来テーブル表示"], horizontal=True, key="hist_view_mode")
            if hist_view_mode == "📱 スマホ用カード表示":
                render_cards_history(df_filtered)
            else:
                st.dataframe(df_filtered, use_container_width=True, height=500)
        else:
            st.info("predictions タブにデータがありません。")
    except Exception as e:
        st.warning(f"読み込みエラー: {e}")


# --- TAB 3: 過去レース一括登録 ＆ タブ全データエクスポート ---
with tab3:
    st.subheader("📥 過去レース結果データ 一括登録 ＆ エクスポート")
    st.caption("スプレッドシートのレース用タブを選択し、データの登録または学習用CSVとしての全件エクスポートが行えます。")

    try:
        sh = get_gspread_client()
        ws_list = [ws.title for ws in sh.worksheets() if ws.title != "horses_master"]
    except Exception:
        ws_list = ["predictions"]

    def_idx = ws_list.index("predictions") if "predictions" in ws_list else 0
    target_ws = st.selectbox("対象タブを選択", ws_list, index=def_idx, key="races_ws")

    with st.expander(f"📥 「{target_ws}」タブの全データをCSV出力する（学習用）", expanded=False):
        if st.button("🔄 対象タブのデータを取得", key="fetch_target_ws"):
            st.cache_data.clear()
        try:
            conn_t = st.connection("gsheets", type=GSheetsConnection)
            df_target_all = conn_t.read(worksheet=target_ws, ttl=0)
            if df_target_all is not None and not df_target_all.empty:
                df_target_all = df_target_all.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
                df_target_clean = ensure_21_columns(df_target_all)
                st.info(f"「{target_ws}」タブから全 **{len(df_target_clean)} 件** のデータを読み込みました。")
                
                filename_target = f"{target_ws}_export_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}.csv"
                link_html_target = create_safe_download_link(df_target_clean, filename_target, f"📥 {target_ws} をCSV保存（スマホ対応）")
                st.markdown(link_html_target, unsafe_allow_html=True)
                st.dataframe(df_target_clean, use_container_width=True, height=200)
            else:
                st.info(f"「{target_ws}」タブにデータがありません。")
        except Exception as e:
            st.warning(f"読み込みエラー: {e}")

    st.markdown("---")
    st.subheader("📝 過去レースデータのテキスト一括登録")
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
                df_horses_hist = df_horses_hist.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
                for c in COLUMNS_HORSES:
                    if c not in df_horses_hist.columns:
                        df_horses_hist[c] = ""
                df_horses_hist = df_horses_hist[COLUMNS_HORSES].fillna("").astype(str)

                st.info(f"全 **{len(df_horses_hist)} 頭** の血統データを読み込みました。")
                
                filename_horses = f"horses_master_export_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}.csv"
                link_html_horses = create_safe_download_link(df_horses_hist, filename_horses, "📥 horses_master をCSV保存（スマホ対応）")
                st.markdown(link_html_horses, unsafe_allow_html=True)

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
            df_h = df_h[COLUMNS_HORSES].fillna("").reset_index(drop=True).astype(str)

            st.success(f"解析成功: **{len(df_h)} 頭** の血統データ（書き込み先: **horses_master**）")
            st.dataframe(df_h, use_container_width=True)

            if st.button("🚀 horses_master タブに書き込む", type="primary"):
                with st.spinner("書き込み中..."):
                    sh = get_gspread_client()
                    worksheet = sh.worksheet("horses_master")
                    clean_horse_rows = df_h.values.tolist()
                    worksheet.append_rows(clean_horse_rows, value_input_option="USER_ENTERED")
                    st.balloons()
                    st.success(f"🎉 正常に {len(df_h)} 頭の血統データが追加されました！")
        except Exception as e:
            st.error(f"データの解析または書き込みに失敗しました: {e}")
