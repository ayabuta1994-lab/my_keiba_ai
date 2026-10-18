# -*- coding: utf-8 -*-
"""
🏇 自分専用 競馬予想AI（オールインワン版・タブ並び順調整済み）
- AI予想実行 & predictionsへの自動保存
- 過去データ参照（AI予想の隣）
- 過去レースデータ一括登録（タブ選択可・血統タブ除外）
- 血統データ一括登録（horses_master専用）
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

st.set_page_config(page_title="競馬予想AI オールインワン", page_icon="🏇", layout="wide")

MODEL_PATH = Path(__file__).parent / "model.pkl"
JST = timezone(timedelta(hours=9))

STD_COLS = [
    "Bracket", "Horse Number", "Horse Name", "Sex", "Horse Age",
    "Burden Weight", "Jockey", "Trainer", "Sire", "Dam's Sire", "Tactics",
    "Total Prize", "Last Placing", "Last Time",
]
NUM_DEFAULTS = {
    "Bracket": 0, "Horse Number": 0, "Horse Age": 4, "Burden Weight": 55,
    "Total Prize": 0, "Last Placing": "median:9", "Last Time": "mean:90",
    "Distance": 0,
}
STR_COLS = ["Horse Name", "Sex", "Jockey", "Trainer", "Sire", "Dam's Sire",
            "Venue", "Surface", "Track Condition", "Weather"]
TACTICS_LABEL = {1: "逃げ", 2: "先行", 3: "差し", 4: "追込"}

VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉", "その他"]
SURFACES = ["芝", "ダート", "障害"]
DISTANCES = sorted(set(list(range(1000, 3700, 200)) + [1150, 1700, 1800, 2100, 2300, 2500, 2600, 2800, 3000]))
TRACKS = ["良", "稍重", "重", "不良"]
WEATHERS = ["晴", "曇", "小雨", "雨", "雪"]

COLUMNS_PREDICTIONS = [
    "Execution Date", "Race Name", "Venue", "Surface", "Distance", 
    "Track Condition", "Weather", "Pace", "High Pace Prob", "Front Ratio",
    "Rank", "Horse Number", "Horse Name", "Sire", "Dam's Sire", "Tactics", "AI Probability"
]

COLUMNS_HORSES = [
    "Horse Name", "Sex", "Birth Year", "Sire", "Dam's Sire", "Trainer", "Owner", "Note"
]


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
    creds_dict = dict(st.secrets["gcp_service_account"])
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)
    spreadsheet_id = st.secrets["spreadsheet_id"]
    return gc.open_by_key(spreadsheet_id)


def parse_csv(text: str) -> pd.DataFrame:
    text = text.strip()
    text = text.replace("```csv", "").replace("```", "").replace("\u3000", " ").strip()
    if not text:
        raise ValueError("出馬表が空です。")
    first = text.splitlines()[0]
    sep = "\t" if "\t" in first else ","
    read_kw = dict(sep=sep, dtype=str, skipinitialspace=True, engine="python", on_bad_lines="skip")

    df = pd.read_csv(StringIO(text), **read_kw)
    df.columns = [str(c).strip() for c in df.columns]
    lookup = {norm(c): c for c in df.columns}
    ren = {lookup[norm(s)]: s for s in STD_COLS if norm(s) in lookup}

    if len(ren) < 8:
        df = pd.read_csv(StringIO(text), header=None, **read_kw)
        if df.shape[1] < len(STD_COLS):
            raise ValueError("列数が足りません。14列（Bracket〜Last Time）のCSVを貼り付けてください。")
        df = df.iloc[:, : len(STD_COLS)]
        df.columns = STD_COLS
    else:
        df = df.rename(columns=ren)

    for c in STD_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[STD_COLS].copy()
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
    df = df.dropna(how="all")
    df = df[~(df["Horse Name"].isna() & df["Horse Number"].isna())]
    return df.reset_index(drop=True)


def parse_time(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip()
    m = re.match(r"^(\d+):(\d+(?:\.\d+)?)$", s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.match(r"^(\d+)[.:](\d{2})[.:](\d+)$", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + float("0." + m.group(3))
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_tactics(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
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


def preprocess(df: pd.DataFrame, venue, surface, distance, track, weather) -> pd.DataFrame:
    df = df.copy()
    df["Venue"] = venue
    df["Surface"] = surface
    df["Distance"] = distance
    df["Track Condition"] = track
    df["Weather"] = weather

    for c in NUM_DEFAULTS:
        if c == "Last Time":
            df[c] = df[c].map(parse_time)
        else:
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


def render_pace(p: dict, out: pd.DataFrame):
    st.subheader("③ 展開予想・ペース診断")
    if not p.get("ok"):
        st.info("脚質（Tactics）が読み取れなかったため、ペース診断は行えませんでした。")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("想定ペース", p["pace"])
    m2.metric("ハイペース確率", f"{p['high_prob']:.1f}%")
    m3.metric("逃げ・先行の割合", f"{p['front_ratio']:.1f}%")


DEFAULT_FEATURES = [
    "Venue", "Surface", "Distance", "Track Condition", "Weather", "Bracket", "Horse Number",
    "Sex", "Horse Age", "Burden Weight", "Jockey", "Trainer", "Sire", "Dam's Sire", "Tactics",
    "Total Prize", "Last Placing", "Last Time",
]


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


# メインタイトル
st.title("🏇 自分専用 競馬予想AI（オールインワン版）")

try:
    model, meta = load_model()
    if model is None:
        st.error("model.pkl が見つからないか、有効なモデルではありません。")
        st.stop()
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")
    st.stop()

# タブの定義（「過去データ参照」を「AI予想の実行」の隣に配置）
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 AI予想の実行", 
    "📊 過去データ参照", 
    "📥 過去レース一括登録", 
    "🧬 血統データ登録"
])

# --- TAB 1: AI予想の実行 ---
with tab1:
    st.subheader("① レース情報の設定")
    c1, c2 = st.columns(2)
    with c1:
        venue = st.selectbox("競馬場（Venue）", VENUES, index=VENUES.index("東京"))
        surface = st.selectbox("芝・ダート（Surface）", SURFACES)
        track = st.selectbox("馬場状態（Track Condition）", TRACKS)
    with c2:
        distance = st.selectbox("距離（Distance）", DISTANCES, index=DISTANCES.index(1600))
        weather = st.selectbox("天気（Weather）", WEATHERS)
    race_name = st.text_input("レース名（任意・記録用）", placeholder="例：天皇賞（秋）")

    st.subheader("② 出馬表の貼り付け")
    st.caption("列順: Bracket, Horse Number, Horse Name, Sex, Horse Age, Burden Weight, "
               "Jockey, Trainer, Sire, **Dam's Sire**, Tactics, Total Prize, Last Placing, Last Time")
    csv_text = st.text_area(
        "出馬表CSV / TSV",
        height=280,
        placeholder=(
            "Bracket,Horse Number,Horse Name,Sex,Horse Age,Burden Weight,Jockey,Trainer,"
            "Sire,Dam's Sire,Tactics,Total Prize,Last Placing,Last Time\n"
            "1,1,サンプルホース,牡,4,58,ルメール,友道,ドゥラメンテ,サンデーサイレンス,2,12000,2,1:34.5"
        ),
    )

    run = st.button("🔮 AI予想を実行する", type="primary", use_container_width=True)

    if run:
        if not csv_text.strip():
            st.warning("出馬表データを貼り付けてください。")
        else:
            try:
                raw = parse_csv(csv_text)
                if raw.empty:
                    raise ValueError("出走馬が1頭も読み取れませんでした。")
                df = preprocess(raw, venue, surface, distance, track, weather)
                pace = diagnose_pace(df)

                feats = get_feature_names_custom(model, meta) or DEFAULT_FEATURES
                X, missing = build_X(df, feats)
                X, hashed = encode_X(X, model, meta)
                proba = predict_top3(model, X)

                now = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
                out = pd.DataFrame({
                    "馬番": df["Horse Number"].astype(int),
                    "馬名": df["Horse Name"],
                    "父": df["Sire"],
                    "母父": df["Dam's Sire"],
                    "想定脚質": df["脚質ラベル"],
                    "AI予想確率(%)": np.round(np.asarray(proba, dtype=float) * 100, 1),
                })
                out = out.sort_values("AI予想確率(%)", ascending=False, kind="stable").reset_index(drop=True)
                out.insert(0, "順位", np.arange(1, len(out) + 1))

                st.session_state["result"] = out
                st.session_state["pace"] = pace
                st.session_state["notes"] = (missing, hashed)

                log = out.copy()
                log.insert(0, "Execution Date", now)
                log.insert(1, "Race Name", race_name)
                log.insert(2, "Venue", venue)
                log.insert(3, "Surface", surface)
                log.insert(4, "Distance", distance)
                log.insert(5, "Track Condition", track)
                log.insert(6, "Weather", weather)
                log.insert(7, "Pace", pace.get("pace", "不明"))
                log.insert(8, "High Pace Prob", pace.get("high_prob", ""))
                log.insert(9, "Front Ratio", pace.get("front_ratio", ""))

                # predictions タブへ自動保存
                sh = get_gspread_client()
                ws = sh.worksheet("predictions")
                ws.append_rows(log.values.tolist(), value_input_option="USER_ENTERED")
                st.session_state["saved"] = True
            except Exception as e:
                st.session_state.pop("result", None)
                st.error(f"予想または保存中にエラーが発生しました: {e}")
                with st.expander("詳細ログ"):
                    st.code(traceback.format_exc())

    if "result" in st.session_state:
        out = st.session_state["result"]
        missing, hashed = st.session_state.get("notes", ([], []))
        render_pace(st.session_state.get("pace", {"ok": False}), out)

        st.subheader(f"④ AI予想結果（全{len(out)}頭・3着以内確率順）")
        st.dataframe(
            out,
            hide_index=True,
            use_container_width=True,
            height=int(35 * (len(out) + 1) + 3),
        )
        if st.session_state.get("saved"):
            st.success("✨ 予想完了＆ `predictions` タブへのデータ蓄積が完了しました！")


# --- TAB 2: 過去データ参照（AI予想の隣） ---
with tab2:
    st.subheader("📊 過去の予想データ参照（predictions タブ）")
    if st.button("🔄 データを再読み込み"):
        st.cache_data.clear()

    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df_hist = conn.read(worksheet="predictions", ttl=0)
        if df_hist is not None and not df_hist.empty:
            df_hist = df_hist.dropna(how="all").dropna(axis=1, how="all")
            st.success(f"全 {len(df_hist)} 行のデータを読み込みました。")
            st.dataframe(df_hist, use_container_width=True, height=500)
        else:
            st.info("predictions タブにデータがありません。")
    except Exception as e:
        st.warning(f"読み込みエラー: {e}")


# --- TAB 3: 過去レース一括登録 ---
with tab3:
    st.subheader("📥 過去レース結果データ 一括登録")
    st.caption("スプレッドシートのレース用タブ（predictions や過去別タブ）を選択して一括登録できます。")

    try:
        sh = get_gspread_client()
        ws_list = [ws.title for ws in sh.worksheets() if ws.title != "horses_master"]
    except Exception:
        ws_list = ["predictions"]

    def_idx = ws_list.index("predictions") if "predictions" in ws_list else 0
    target_ws = st.selectbox("書き込み先タブを選択", ws_list, index=def_idx, key="races_ws")

    raw_races = st.text_area(
        "過去レースデータの貼り付け（TSV / CSV）",
        height=220,
        key="races_text",
        placeholder="Execution Date\tRace Name\tVenue..."
    )

    if raw_races.strip():
        try:
            sep = '\t' if '\t' in raw_races else ','
            df_r = pd.read_csv(io.StringIO(raw_races), sep=sep, dtype=str)
            for col in COLUMNS_PREDICTIONS:
                if col not in df_r.columns:
                    df_r[col] = ""
            df_r = df_r[COLUMNS_PREDICTIONS].fillna("")

            st.success(f"解析成功: **{len(df_r)} 行**（書き込み先: **{target_ws}**）")
            st.dataframe(df_r, use_container_width=True)

            if st.button(f"🚀 「{target_ws}」タブに一括書き込み", type="primary"):
                with st.spinner("書き込み中..."):
                    worksheet = sh.worksheet(target_ws)
                    worksheet.append_rows(df_r.values.tolist(), value_input_option="USER_ENTERED")
                    st.balloons()
                    st.success(f"🎉 正常に {len(df_r)} 件のデータが追加されました！")
        except Exception as e:
            st.error(f"データの解析または書き込みに失敗しました: {e}")


# --- TAB 4: 血統データ登録 ---
with tab4:
    st.subheader("🧬 血統・馬マスタデータ一括登録（horses_master 専用）")
    st.caption("競走馬の血統マスタデータを `horses_master` タブへ一括登録します。")

    raw_horses = st.text_area(
        "血統データの貼り付け（TSV / CSV）",
        height=220,
        key="horses_text",
        placeholder="Horse Name\tSex\tBirth Year\tSire\tDam's Sire\tTrainer\tOwner\tNote\nサンプルホース\t牡\t2020\tドゥラメンテ\tサンデーサイレンス\t友道\tサンデーR\t特記事項"
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
                    worksheet.append_rows(df_h.values.tolist(), value_input_option="USER_ENTERED")
                    st.balloons()
                    st.success(f"🎉 正常に {len(df_h)} 頭の血統データが追加されました！")
        except Exception as e:
            st.error(f"データの解析または書き込みに失敗しました: {e}")


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
