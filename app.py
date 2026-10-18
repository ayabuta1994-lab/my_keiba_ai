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

RACE_MANAGEMENT_SHEET = "race_management"
RESULTS_SHEET = "race_results"
RESULT_COLUMNS = ["Race ID", "Race Day", "Venue", "Race Name", "Distance", "Horse Number", "Horse Name", "Tactics", "Last 3F", "Placing", "Updated At"]
RACE_MANAGEMENT_COLUMNS = ["Race ID", "Race Day", "Venue", "Race Name", "Distance", "Status", "Horse Count", "Registered At"]


def _race_id(race_day, venue, race_name, distance=""):
    """Stable ID kept outside the 21-column training dataset."""
    parts = [str(race_day).strip(), str(venue).strip(), str(race_name).strip(), str(distance).strip()]
    slug = "_".join(re.sub(r"[^\w一-龯ぁ-んァ-ン]+", "", x)[:24] for x in parts)
    return slug or f"race_{datetime.now(JST).strftime('%Y%m%d%H%M%S')}"


def _ensure_race_management_sheet(spreadsheet):
    try:
        ws = spreadsheet.worksheet(RACE_MANAGEMENT_SHEET)
    except Exception:
        ws = spreadsheet.add_worksheet(title=RACE_MANAGEMENT_SHEET, rows=1000, cols=len(RACE_MANAGEMENT_COLUMNS))
        ws.append_row(RACE_MANAGEMENT_COLUMNS, value_input_option="USER_ENTERED")
        return ws
    values = _cached_sheet_values(spreadsheet.id, ws.title)
    if not values:
        ws.append_row(RACE_MANAGEMENT_COLUMNS, value_input_option="USER_ENTERED")
    elif [str(v).strip() for v in values[0][:len(RACE_MANAGEMENT_COLUMNS)]] != RACE_MANAGEMENT_COLUMNS:
        raise ValueError(f"「{RACE_MANAGEMENT_SHEET}」タブのヘッダーが想定と異なります。既存データを保護するため自動変更しません。")
    return ws


def _register_race(spreadsheet, df):
    """Register race metadata without adding Race ID to COLUMNS_21."""
    if df is None or df.empty:
        return ""
    first = df.iloc[0]
    race_day = str(first.get("Race Day", "")).strip()
    venue = str(first.get("Venue", "")).strip()
    race_name = str(first.get("Race Name", "")).strip()
    distance = str(first.get("Distance", "")).strip()
    if not race_day or not venue or not race_name:
        return ""
    rid = _race_id(race_day, venue, race_name, distance)
    ws = _ensure_race_management_sheet(spreadsheet)
    values = _cached_sheet_values(spreadsheet.id, ws.title)
    headers = values[0] if values else RACE_MANAGEMENT_COLUMNS
    existing = values[1:] if len(values) > 1 else []
    for row in existing:
        row = row + [""] * (len(headers) - len(row))
        if row[0] == rid:
            return rid
    ws.append_row([rid, race_day, venue, race_name, distance, "予想登録済み", str(len(df)), datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")], value_input_option="USER_ENTERED")
    _cached_sheet_values.clear()
    return rid


def _ensure_results_sheet(spreadsheet):
    """Actual post-race results are stored separately from immutable prediction rows."""
    try:
        ws = spreadsheet.worksheet(RESULTS_SHEET)
    except Exception:
        ws = spreadsheet.add_worksheet(title=RESULTS_SHEET, rows=5000, cols=len(RESULT_COLUMNS))
        ws.append_row(RESULT_COLUMNS, value_input_option="USER_ENTERED")
        return ws
    values = _cached_sheet_values(spreadsheet.id, ws.title)
    if not values:
        ws.append_row(RESULT_COLUMNS, value_input_option="USER_ENTERED")
    elif [str(v).strip() for v in values[0][:len(RESULT_COLUMNS)]] != RESULT_COLUMNS:
        raise ValueError(f"「{RESULTS_SHEET}」タブのヘッダーが想定と異なります。既存データを保護するため自動変更しません。")
    return ws


def _previous_result_lookup(spreadsheet, race_day):
    """Return each horse's latest known result strictly before the target race day."""
    try:
        ws = spreadsheet.worksheet(RESULTS_SHEET)
        df = _read_worksheet_dataframe(ws)
    except Exception:
        return {}
    if df.empty or not all(c in df.columns for c in ("Race Day", "Horse Name", "Tactics", "Last 3F")):
        return {}
    target = pd.to_datetime(race_day, errors="coerce")
    df = df.copy()
    df["_date"] = pd.to_datetime(df["Race Day"], errors="coerce")
    if pd.notna(target):
        df = df[df["_date"] < target]
    df["_horse"] = df["Horse Name"].map(_norm_horse)
    df = df.sort_values("_date")
    lookup = {}
    for _, row in df.iterrows():
        if row["_horse"]:
            lookup[row["_horse"]] = {"Tactics": parse_tactics(row.get("Tactics", "")), "Last 3F": row.get("Last 3F", "")}
    return lookup


def _load_race_management(spreadsheet):
    ws = _ensure_race_management_sheet(spreadsheet)
    values = _cached_sheet_values(spreadsheet.id, ws.title)
    # One-time/backfill behavior: make older races already stored in predictions selectable too.
    try:
        pred_ws = spreadsheet.worksheet("predictions")
        pred_df = _read_worksheet_dataframe(pred_ws)
        if not pred_df.empty and all(c in pred_df.columns for c in ("Race Day", "Venue", "Race Name", "Distance", "Horse Name")):
            clean = ensure_21_columns(pred_df)
            clean = clean[(clean["Race Day"].str.strip() != "") & (clean["Venue"].str.strip() != "") & (clean["Race Name"].str.strip() != "")]
            known_ids = {str(r[0]).strip() for r in values[1:] if r and str(r[0]).strip()}
            seen = set()
            for _, race_row in clean.iterrows():
                rid = _race_id(race_row["Race Day"], race_row["Venue"], race_row["Race Name"], race_row["Distance"])
                if rid in known_ids or rid in seen:
                    continue
                same_race = clean[(clean["Race Day"] == race_row["Race Day"]) & (clean["Venue"] == race_row["Venue"]) & (clean["Race Name"] == race_row["Race Name"]) & (clean["Distance"] == race_row["Distance"])]
                placings = same_race["Placing"].astype(str).str.strip()
                status = "結果登録済み" if placings.ne("").any() else "予想登録済み"
                ws.append_row([rid, race_row["Race Day"], race_row["Venue"], race_row["Race Name"], race_row["Distance"], status, str(len(same_race)), datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")], value_input_option="USER_ENTERED")
                known_ids.add(rid)
                seen.add(rid)
            values = _cached_sheet_values(spreadsheet.id, ws.title)
    except Exception:
        # Keep the result screen usable for manually registered rows even if old predictions cannot be read.
        pass
    if len(values) < 2:
        return ws, pd.DataFrame(columns=RACE_MANAGEMENT_COLUMNS)
    headers = values[0]
    rows = [r + [""] * (len(headers) - len(r)) for r in values[1:] if any(str(x).strip() for x in r)]
    df = pd.DataFrame(rows, columns=headers)
    for c in RACE_MANAGEMENT_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return ws, df[RACE_MANAGEMENT_COLUMNS].fillna("").astype(str)


def _race_display_label(row):
    return f"{row.get('Race Day', '')}｜{row.get('Venue', '')}｜{row.get('Race Name', '')}（{row.get('Distance', '')}m）"


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


@st.cache_data(ttl=30, show_spinner=False)
def _cached_sheet_values(spreadsheet_id, worksheet_title):
    """Cache Google Sheets reads briefly to avoid 429s during Streamlit reruns."""
    sh = get_gspread_client()
    return sh.worksheet(worksheet_title).get_all_values()


@st.cache_data(ttl=30, show_spinner=False)
def _cached_worksheet_titles(spreadsheet_id):
    """Cache worksheet metadata to avoid repeated spreadsheet metadata requests."""
    sh = get_gspread_client()
    return [ws.title for ws in sh.worksheets()]


def _read_worksheet_dataframe(ws):
    values = _cached_sheet_values(ws.spreadsheet.id, ws.title)
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
    for title in _cached_worksheet_titles(spreadsheet.id):
        if title in ("horses_master", RACE_MANAGEMENT_SHEET, RESULTS_SHEET):
            continue
        ws = spreadsheet.worksheet(title)
        if ws.title in ("horses_master", RACE_MANAGEMENT_SHEET, RESULTS_SHEET):
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
            # Explicitly entered previous-race style takes priority; if left blank,
            # fall back to the latest known historical style, otherwise 0 (unknown).
            "Hist_Last_Tactics": (
                float(parse_tactics(r.get("Previous_Tactics_Input")))
                if pd.notna(parse_tactics(r.get("Previous_Tactics_Input")))
                else (float(past["_tactics"].dropna().iloc[-1]) if past["_tactics"].notna().any() else 0.0)
            ),
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


def parse_input_table(raw_text: str) -> pd.DataFrame:
    """21列のCSV/TSVテキストを読み込み、統一した列順のDataFrameを返す。"""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("入力データが空です。競馬データを貼り付けてください。")

    # Excelやスマートフォンからの貼り付けで混在する改行・BOMを整理する。
    text = raw_text.strip().lstrip("\ufeff")
    first_line = text.splitlines()[0]
    delimiter = "\t" if "\t" in first_line else ","

    try:
        parsed = pd.read_csv(
            StringIO(text),
            sep=delimiter,
            header=None,
            dtype=str,
            keep_default_na=False,
            skip_blank_lines=True,
            engine="python",
        )
    except Exception as exc:
        raise ValueError(f"CSV/TSVを解析できませんでした。区切り文字や引用符を確認してください。詳細: {exc}") from exc

    if parsed.empty:
        raise ValueError("読み取れるデータ行がありません。")

    # 先頭行が21列の正式なヘッダーなら除外する。ヘッダーなし形式も受け付ける。
    first_values = [str(v).strip().lstrip("\ufeff") for v in parsed.iloc[0].tolist()]
    normalized = [norm(v) for v in first_values]
    expected = [norm(v) for v in COLUMNS_21]
    has_header = len(first_values) == len(COLUMNS_21) and normalized == expected
    if has_header:
        parsed = parsed.iloc[1:].reset_index(drop=True)

    if parsed.empty:
        raise ValueError("ヘッダー行しかありません。競走馬のデータ行も貼り付けてください。")

    if len(parsed.columns) != len(COLUMNS_21):
        raise ValueError(
            f"列数が不正です。21列必要ですが、{len(parsed.columns)}列検出されました。"
            "カンマ区切りCSVまたはタブ区切りTSVを貼り付けてください。"
        )

    parsed.columns = COLUMNS_21
    return ensure_21_columns(parsed)


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

# --- TAB 1: AI予想の実行（必要時のみスプレッドシートを読む） ---
with tab1:
    st.subheader("① 予想対象レースの選択（Googleスプレッドシート連動）")
    st.caption("スプレッドシートは、利用者が読み込み操作をしたときだけ読み込みます。")

    if st.button("📋 出走表タブ一覧を読み込む", key="load_eval_tabs"):
        try:
            sh = get_gspread_client()
            st.session_state["eval_ws_list"] = [title for title in _cached_worksheet_titles(sh.id) if title not in ("horses_master", RACE_MANAGEMENT_SHEET, RESULTS_SHEET)]
            st.session_state.pop("eval_preview_df", None)
            st.session_state.pop("eval_preview_ws", None)
            st.success("タブ一覧を読み込みました。予想対象を選択してください。")
        except Exception as exc:
            st.error(f"タブ一覧の読み込みに失敗しました: {exc}")

    ws_list = st.session_state.get("eval_ws_list", [])
    if not ws_list:
        st.info("まだスプレッドシートを読み込んでいません。「出走表タブ一覧を読み込む」を押してください。")
    else:
        target_eval_ws = st.selectbox("予想するレースのタブを選択", ws_list, key="eval_ws_select")
        if st.button("📥 選択した出走表を読み込む", key="load_eval_entry"):
            try:
                conn_preview = st.connection("gsheets", type=GSheetsConnection)
                df_preview = conn_preview.read(worksheet=target_eval_ws, ttl=30)
                st.session_state["eval_preview_df"] = df_preview
                st.session_state["eval_preview_ws"] = target_eval_ws
                st.session_state.pop("result_df", None)
                st.success(f"「{target_eval_ws}」を読み込みました。")
            except Exception as exc:
                st.error(f"出走表の読み込みに失敗しました: {exc}")

        df_preview = st.session_state.get("eval_preview_df") if st.session_state.get("eval_preview_ws") == target_eval_ws else None
        sh = None
        edited_tactics = None
        if df_preview is not None and not df_preview.empty:
            preview_df = ensure_21_columns(df_preview).copy()
            st.markdown("**② 出走馬の前走脚質を入力**")
            st.caption("脚質は今回のレースでの位置取りではなく、各馬の前走の脚質です。1=逃げ、2=先行、3=差し、4=追込。分からない馬は空欄のままにしてください。")
            if st.button("🔎 結果管理シートから前走実績を読み込む", key="load_previous_results"):
                try:
                    sh_prev = get_gspread_client()
                    st.session_state["previous_results_lookup"] = _previous_result_lookup(sh_prev, preview_df.iloc[0].get("Race Day", ""))
                    st.session_state["previous_results_for_ws"] = target_eval_ws
                    st.success("前走実績を読み込みました。")
                except Exception as exc:
                    st.warning(f"前走実績の読み込みに失敗しました: {exc}")
            previous_results = st.session_state.get("previous_results_lookup", {}) if st.session_state.get("previous_results_for_ws") == target_eval_ws else {}
            def previous_value(horse_name, field, fallback=""):
                result = previous_results.get(_norm_horse(horse_name), {})
                value = result.get(field, "")
                return value if value not in (None, "", np.nan) else fallback
            tactics_editor = pd.DataFrame({
                "Horse Number": preview_df["Horse Number"].astype(str),
                "Horse Name": preview_df["Horse Name"].astype(str),
                "Tactics": [previous_value(name, "Tactics", parse_tactics(preview_df.iloc[i]["Tactics"])) for i, name in enumerate(preview_df["Horse Name"].astype(str))],
                "Last 3F": [previous_value(name, "Last 3F", preview_df.iloc[i]["Last 3F"]) for i, name in enumerate(preview_df["Horse Name"].astype(str))],
            })
            edited_tactics = st.data_editor(
                tactics_editor, hide_index=True, use_container_width=True,
                key=f"prev_tactics_editor_{target_eval_ws}",
                column_config={
                    "Horse Number": st.column_config.TextColumn("馬番", disabled=True),
                    "Horse Name": st.column_config.TextColumn("馬名", disabled=True),
                    "Tactics": st.column_config.SelectboxColumn("前走脚質コード", options=[1, 2, 3, 4], required=False,
                        help="結果管理シートに前走実績があれば自動入力。1=逃げ / 2=先行 / 3=差し / 4=追込"),
                    "Last 3F": st.column_config.TextColumn("前走上がり3F", help="前走実績から自動入力。必要に応じて修正できます。"),
                }, disabled=["Horse Number", "Horse Name"])

        run = st.button("🔮 前走脚質を反映してAI予想を実行する", type="primary", use_container_width=True, disabled=(df_preview is None or df_preview.empty or edited_tactics is None))
        if run:
            try:
                sh = get_gspread_client()
                raw_df = ensure_21_columns(df_preview).copy()
                raw_df["Previous_Tactics_Input"] = edited_tactics["Tactics"].map(parse_tactics).values
                raw_df["Tactics"] = edited_tactics["Tactics"].values
                raw_df["Last 3F"] = edited_tactics["Last 3F"].values
                source_ws = sh.worksheet(target_eval_ws)
                source_headers = [str(v).strip() for v in source_ws.row_values(1)]
                if "Tactics" not in source_headers:
                    raise ValueError("選択した出走表に Tactics 列がありません。21列の固定ヘッダーを確認してください。")
                tactics_col = source_headers.index("Tactics") + 1
                tactics_col_letter = gspread.utils.rowcol_to_a1(1, tactics_col).rstrip("1")
                tactics_values = [["" if pd.isna(v) else int(v)] for v in raw_df["Tactics"]]
                if tactics_values:
                    source_ws.update(range_name=f"{tactics_col_letter}2:{tactics_col_letter}{len(tactics_values)+1}", values=tactics_values, value_input_option="USER_ENTERED")
                st.session_state.pop("top3_probabilities", None)
                now_str = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
                df = preprocess(raw_df, {"Execution Date": now_str})
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
                    feats = get_feature_names_custom(model, meta) or ["Bracket", "Horse Number", "Horse Age", "Burden Weight", "Tactics", "Distance", "Last 3F"]
                    X, missing = build_X(df, feats)
                    X, hashed = encode_X(X, model, meta)
                    proba = predict_top3(model, X)
                proba_pct = np.round(np.asarray(proba, dtype=float) * 100, 1)
                df["AI_Probability"] = proba_pct.astype(str)
                df["AI_Rank"] = pd.Series(proba_pct, index=df.index).rank(ascending=False, method="min").astype(int).astype(str)
                if top3_pct is not None:
                    df["複勝確率(3着以内)"] = top3_pct.astype(str)
                df_sorted = df.sort_values("AI_Probability", key=lambda col: pd.to_numeric(col, errors="coerce"), ascending=False).reset_index(drop=True)
                out_21 = ensure_21_columns(df_sorted)
                st.session_state["result_df"] = out_21
                st.session_state["result_view_df"] = df_sorted.copy()
                st.session_state["pace"] = pace
                ws = sh.worksheet("predictions")
                ws.append_rows(out_21.values.tolist(), value_input_option="USER_ENTERED")
                _cached_sheet_values.clear(); _cached_worksheet_titles.clear()
                st.session_state["registered_race_id"] = _register_race(sh, out_21)
                st.session_state["saved"] = True
                st.success("予測結果を保存しました。")
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
            if view_mode == "📱 スマホ用カード表示": render_cards_prediction(view_out)
            else: st.dataframe(view_out, hide_index=True, use_container_width=True, height=int(35 * (len(view_out) + 1) + 3))
            if st.session_state.get("saved"):
                st.success("✨ 予想完了＆ `predictions` タブへの21列データ蓄積が完了しました！")
                rid = st.session_state.get("registered_race_id", "")
                if rid: st.caption(f"結果入力用レースID（21列の外で管理）: `{rid}`")

# --- TAB 2: 過去データ参照 ---
with tab2:
    st.subheader("📊 過去の予想データ参照（predictions タブ）")
    if st.button("📥 predictions を読み込む", key="load_predictions_history"):
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            st.session_state["predictions_history_df"] = conn.read(worksheet="predictions", ttl=30)
            st.success("predictions を読み込みました。")
        except Exception as e:
            st.warning(f"読み込みエラー: {e}")
    if st.button("🔄 キャッシュを破棄して再読み込み", key="reload_predictions_history"):
        _cached_sheet_values.clear(); _cached_worksheet_titles.clear(); st.cache_data.clear()
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            st.session_state["predictions_history_df"] = conn.read(worksheet="predictions", ttl=0)
            st.success("最新データを読み込みました。")
        except Exception as e:
            st.warning(f"読み込みエラー: {e}")
    df_hist_raw = st.session_state.get("predictions_history_df")
    if df_hist_raw is None:
        st.info("まだデータを読み込んでいません。表示したいときに「predictions を読み込む」を押してください。")
    else:
        try:
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

    if st.button("📋 登録先タブ一覧を読み込む", key="load_import_tabs"):
        try:
            sh = get_gspread_client()
            st.session_state["import_ws_list"] = [title for title in _cached_worksheet_titles(sh.id) if title not in ("horses_master", RACE_MANAGEMENT_SHEET, RESULTS_SHEET)]
        except Exception as exc:
            st.error(f"タブ一覧の読み込みに失敗しました: {exc}")
    ws_list = st.session_state.get("import_ws_list", [])
    target_ws = None
    if ws_list:
        def_idx = ws_list.index("predictions") if "predictions" in ws_list else 0
        target_ws = st.selectbox("対象タブを選択", ws_list, index=def_idx, key="races_ws")
        if st.button("📥 選択したタブの全データを取得", key="fetch_target_ws"):
            try:
                conn_t = st.connection("gsheets", type=GSheetsConnection)
                st.session_state["target_export_df"] = conn_t.read(worksheet=target_ws, ttl=30)
                st.session_state["target_export_ws"] = target_ws
            except Exception as exc:
                st.error(f"読み込みエラー: {exc}")
        if st.session_state.get("target_export_ws") == target_ws and st.session_state.get("target_export_df") is not None:
            df_target_all = st.session_state["target_export_df"]
            if df_target_all is not None and not df_target_all.empty:
                df_target_all = df_target_all.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
                df_target_clean = ensure_21_columns(df_target_all)
                st.info(f"「{target_ws}」タブから全 **{len(df_target_clean)} 件** のデータを読み込みました。")
                filename_target = f"{target_ws}_export_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}.csv"
                st.markdown(create_safe_download_link(df_target_clean, filename_target, f"📥 {target_ws} をCSV保存（スマホ対応）"), unsafe_allow_html=True)
                st.dataframe(df_target_clean, use_container_width=True, height=200)
            else:
                st.info(f"「{target_ws}」タブにデータがありません。")
    else:
        st.info("スプレッドシートを読み込んでいません。「登録先タブ一覧を読み込む」を押してください。")

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

            if target_ws and st.button(f"🚀 「{target_ws}」タブに21列データを一括書き込み", type="primary"):
                with st.spinner("書き込み中..."):
                    sh_write = get_gspread_client()
                    worksheet = sh_write.worksheet(target_ws)
                    clean_rows = df_r.fillna("").astype(str).values.tolist()
                    worksheet.append_rows(clean_rows, value_input_option="USER_ENTERED")
                    _cached_sheet_values.clear()
                    _cached_worksheet_titles.clear()
                    st.balloons()
                    st.success(f"🎉 正常に {len(df_r)} 件のデータが追加されました！")
        except Exception as e:
            st.error(f"データの解析または書き込みに失敗しました: {e}")


    st.markdown("---")
    st.subheader("🏁 レースを選択して結果を登録")
    st.caption("実際の脚質・上がり3F・着順は race_results に保存します。predictions の21列とAI予測値は変更しません。次走の予測入力では、日付が前の最新結果を前走実績として自動参照します。")
    if st.button("📋 結果登録対象のレース一覧を読み込む", key="load_result_races"):
        try:
            sh_results = get_gspread_client()
            registry_ws, registry_df = _load_race_management(sh_results)
            st.session_state["race_registry_df"] = registry_df
            st.session_state["race_registry_loaded"] = True
        except Exception as exc:
            st.error(f"レース一覧の読み込みに失敗しました: {exc}")
    if not st.session_state.get("race_registry_loaded"):
        st.info("レース一覧は自動では読み込みません。結果登録を行うときに上のボタンを押してください。")
    else:
        try:
            sh_results = get_gspread_client()
            registry_ws = sh_results.worksheet(RACE_MANAGEMENT_SHEET)
            registry_df = st.session_state.get("race_registry_df", pd.DataFrame(columns=RACE_MANAGEMENT_COLUMNS))
            if registry_df.empty:
                st.info("まだレース管理データがありません。先に出走表からAI予想を実行してください。")
            else:
                labels = [_race_display_label(row) for _, row in registry_df.iterrows()]
                label_to_index = {label: i for i, label in enumerate(labels)}
                chosen_label = st.selectbox("結果を登録するレース", labels, key="result_race_select")
                chosen = registry_df.iloc[label_to_index[chosen_label]]
                st.caption(f"管理用レースID: `{chosen['Race ID']}` ／ 状態: {chosen['Status'] or '未登録'}")
                result_text = st.text_area(
                    "結果データ（TSV / CSV）",
                    height=200,
                    key="race_result_text",
                    placeholder="Horse Number\tHorse Name\tTactics\tLast 3F\tPlacing\n1\tサンプルホースA\t2\t34.5\t1\n2\tサンプルホースB\t3\t35.1\t2"
                )
                if result_text.strip():
                    sep = "\t" if "\t" in result_text.splitlines()[0] else ","
                    result_df = pd.read_csv(StringIO(result_text.strip().lstrip("\ufeff")), sep=sep, dtype=str, keep_default_na=False, engine="python")
                    result_df.columns = [str(c).strip() for c in result_df.columns]
                    aliases = {"馬番": "Horse Number", "馬名": "Horse Name", "脚質": "Tactics", "前走脚質": "Tactics", "上がり3F": "Last 3F", "上がり": "Last 3F", "着順": "Placing", "順位": "Placing"}
                    result_df = result_df.rename(columns={c: aliases.get(c, c) for c in result_df.columns})
                    required = ["Horse Number", "Horse Name", "Tactics", "Last 3F", "Placing"]
                    if not all(c in result_df.columns for c in required):
                        st.error("ヘッダーは Horse Number / Horse Name / Tactics / Last 3F / Placing（または 馬番 / 馬名 / 脚質 / 上がり3F / 着順）にしてください。")
                    else:
                        result_df = result_df[required].fillna("").astype(str)
                        st.dataframe(result_df, use_container_width=True)
                        if st.button("選択したレースの実績を保存", type="primary", key="apply_race_results"):
                            results_ws = _ensure_results_sheet(sh_results)
                            current = _cached_sheet_values(sh_results.id, results_ws.title)
                            headers = current[0]
                            idx = {c: headers.index(c) for c in RESULT_COLUMNS}
                            existing = current[1:]
                            row_map = {}
                            for sheet_row_num, row in enumerate(existing, start=2):
                                row = row + [""] * max(0, len(headers) - len(row))
                                key = (row[idx["Race ID"]], row[idx["Horse Number"]], _norm_horse(row[idx["Horse Name"]]))
                                row_map[key] = sheet_row_num
                            updates, appends = [], []
                            now = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
                            for _, rr in result_df.iterrows():
                                if not rr["Horse Number"].strip() or not rr["Horse Name"].strip():
                                    continue
                                key = (str(chosen["Race ID"]), rr["Horse Number"].strip(), _norm_horse(rr["Horse Name"]))
                                values = [str(chosen["Race ID"]), str(chosen["Race Day"]), str(chosen["Venue"]), str(chosen["Race Name"]), str(chosen["Distance"]), rr["Horse Number"].strip(), rr["Horse Name"].strip(), rr["Tactics"].strip(), rr["Last 3F"].strip(), rr["Placing"].strip(), now]
                                if key in row_map:
                                    updates.append({"range": f"A{row_map[key]}:K{row_map[key]}", "values": [values]})
                                else:
                                    appends.append(values)
                            if not updates and not appends:
                                st.error("保存対象がありません。馬番と馬名を確認してください。")
                            else:
                                if updates:
                                    results_ws.batch_update(updates, value_input_option="USER_ENTERED")
                                if appends:
                                    results_ws.append_rows(appends, value_input_option="USER_ENTERED")
                                _cached_sheet_values.clear()
                                _cached_worksheet_titles.clear()
                                reg_values = _cached_sheet_values(sh_results.id, registry_ws.title)
                                reg_headers = reg_values[0]
                                rid_col = reg_headers.index("Race ID")
                                status_col = reg_headers.index("Status")
                                for reg_row_num, reg_row in enumerate(reg_values[1:], start=2):
                                    reg_row = reg_row + [""] * max(0, len(reg_headers) - len(reg_row))
                                    if reg_row[rid_col] == chosen["Race ID"]:
                                        status_letter = gspread.utils.rowcol_to_a1(1, status_col + 1).rstrip("1")
                                        registry_ws.update_acell(f"{status_letter}{reg_row_num}", "結果登録済み")
                                        _cached_sheet_values.clear()
                                        break
                                st.success(f"実績を保存しました（更新 {len(updates)} 行・新規 {len(appends)} 行）。predictions の21列とAI予測値は変更していません。")
        except Exception as exc:
            st.warning(f"レース結果入力画面の読み込みエラー: {exc}")
# --- TAB 4: 血統データ登録 ＆ CSVダウンロード ---
with tab4:
    st.subheader("🧬 血統データ管理（horses_master 専用）")
    st.caption("血統データ（馬名・父・母父）の登録および学習用CSVダウンロードが行えます。")

    with st.expander("📥 登録済みの血統データをCSVダウンロード", expanded=False):
        if st.button("📥 血統データを読み込む", key="load_horses_history"):
            try:
                conn_h = st.connection("gsheets", type=GSheetsConnection)
                st.session_state["horses_history_df"] = conn_h.read(worksheet="horses_master", ttl=30)
            except Exception as exc:
                st.warning(f"血統データの読み込みエラー: {exc}")
        df_horses_hist = st.session_state.get("horses_history_df")
        if df_horses_hist is not None:
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
