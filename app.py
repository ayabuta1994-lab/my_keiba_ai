# -*- coding: utf-8 -*-
"""
🏇 自分専用 競馬予想AI（v5）× 展開・血統

必要ファイル（GitHubリポジトリ直下）:
  - app.py / requirements.txt / model.pkl

Streamlit Cloud の Secrets（Settings > Secrets）に以下を設定:
  [connections.gsheets]
  spreadsheet = "https://docs.google.com/spreadsheets/d/xxxxxxxx/edit"
  type = "service_account"
  project_id = "..."
  private_key_id = "..."
  private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
  client_email = "xxx@xxx.iam.gserviceaccount.com"
  client_id = "..."
  auth_uri = "https://accounts.google.com/o/oauth2/auth"
  token_uri = "https://oauth2.googleapis.com/token"
  auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
  client_x509_cert_url = "..."

スプレッドシートに "predictions" シートを作り、サービスアカウントのメールを「編集者」で共有してください。
"""
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
from streamlit_gsheets import GSheetsConnection

st.set_page_config(page_title="競馬予想AI v5", page_icon="🏇", layout="centered")

MODEL_PATH = Path(__file__).parent / "model.pkl"
JST = timezone(timedelta(hours=9))

STD_COLS = [
    "Bracket", "Horse Number", "Horse Name", "Sex", "Horse Age",
    "Burden Weight", "Jockey", "Trainer", "Sire", "Tactics",
    "Total Prize", "Last Placing", "Last Time",
]
NUM_DEFAULTS = {  # 数値列: 欠損時の補完方法（Tactics は別処理）
    "Bracket": 0, "Horse Number": 0, "Horse Age": 4, "Burden Weight": 55,
    "Total Prize": 0, "Last Placing": "median:9", "Last Time": "mean:90",
    "Distance": 0,
}
STR_COLS = ["Horse Name", "Sex", "Jockey", "Trainer", "Sire",
            "Venue", "Track Condition", "Weather"]
TACTICS_LABEL = {1: "逃げ", 2: "先行", 3: "差し", 4: "追込"}

VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉", "その他"]
DISTANCES = sorted(set(list(range(1000, 3700, 200)) + [1150, 1700, 1800, 2100, 2300, 2500, 2600, 2800, 3000]))
TRACKS = ["良", "稍重", "重", "不良"]
WEATHERS = ["晴", "曇", "小雨", "雨", "雪"]


def norm(s) -> str:
    return re.sub(r"[\s_\-]", "", str(s)).lower()


# ------------------------------------------------------------------
# モデル読み込み
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="AIの頭脳を読み込み中…")
def load_model():
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

    if model is None or not hasattr(model, "predict"):
        raise ValueError("model.pkl の中に predict() を持つモデルが見つかりませんでした。")
    return model, meta


def get_feature_names(model, meta):
    if meta.get("features"):
        return meta["features"]
    for attr in ("feature_names_in_", "feature_name_", "feature_names_", "feature_names"):
        v = getattr(model, attr, None)
        if v is not None and not callable(v):
            try:
                return list(v)
            except TypeError:
                pass
    booster = getattr(model, "booster_", None)
    if booster is not None:
        try:
            return list(booster.feature_name())
        except Exception:
            pass
    if hasattr(model, "get_booster"):
        try:
            names = model.get_booster().feature_names
            if names:
                return list(names)
        except Exception:
            pass
    if hasattr(model, "feature_name"):
        try:
            return list(model.feature_name())
        except Exception:
            pass
    return None


# ------------------------------------------------------------------
# 入力データの解析・補正
# ------------------------------------------------------------------
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

    if len(ren) < 8:  # ヘッダー無しとみなす
        df = pd.read_csv(StringIO(text), header=None, **read_kw)
        if df.shape[1] < len(STD_COLS):
            raise ValueError("列数が足りません。13列（Bracket〜Last Time）のCSVを貼り付けてください。")
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
    """脚質を 1=逃げ 2=先行 3=差し 4=追込 に正規化。"""
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


def preprocess(df: pd.DataFrame, venue, distance, track, weather) -> pd.DataFrame:
    df = df.copy()
    df["Venue"] = venue
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

    # 脚質: 表示用ラベルと、モデル入力用の数値
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


# ------------------------------------------------------------------
# 展開予想・ペース診断
# ------------------------------------------------------------------
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

    breakdown = (f"逃げ{p['n_nige']}頭 / 先行{p['n_senko']}頭 / "
                 f"差し{p['n_sashi']}頭 / 追込{p['n_oikomi']}頭")

    if p["pace"] == "ハイペース":
        st.warning(f"🔥 **ハイペース想定**（ハイペース確率 {p['high_prob']:.1f}%）\n\n"
                   "逃げ・先行馬が多く、前が競り合って流れが速くなりそうです。"
                   "**差し馬が有利になりそう**で、前に行く馬は最後に脚が鈍るリスクがあります。\n\n"
                   f"脚質内訳: {breakdown}")
        favored, tag = ["差し", "追込"], "差し・追込"
    elif p["pace"] == "スローペース":
        st.info(f"🐢 **スローペース想定**（ハイペース確率 {p['high_prob']:.1f}%）\n\n"
                "前に行きたい馬が少なく、流れが落ち着きそうです。"
                "**前残りの展開になりそう**で、逃げ・先行馬が粘り込みやすい一方、"
                "後方の馬は展開面で不利になりがちです。\n\n"
                f"脚質内訳: {breakdown}")
        favored, tag = ["逃げ", "先行"], "逃げ・先行"
    else:
        st.info(f"⚖️ **ミドルペース想定**（ハイペース確率 {p['high_prob']:.1f}%）\n\n"
                "極端な前傾・後傾にはなりにくく、脚質より馬の能力が結果に出やすい展開です。\n\n"
                f"脚質内訳: {breakdown}")
        favored, tag = None, None

    if favored:
        picks = out[out["想定脚質"].isin(favored)].head(3)
        if len(picks):
            names = "、".join(f"{int(r['馬番'])}番 {r['馬名']}（{r['AI予想確率(%)']:.1f}%）"
                              for _, r in picks.iterrows())
            st.caption(f"📌 展開が向きそうな{tag}で確率上位: {names}")


# ------------------------------------------------------------------
# 特徴量の組み立て・型エラー強制補正
# ------------------------------------------------------------------
DEFAULT_FEATURES = [
    "Venue", "Distance", "Track Condition", "Weather", "Bracket", "Horse Number",
    "Sex", "Horse Age", "Burden Weight", "Jockey", "Trainer", "Sire", "Tactics",
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


def predict_top3(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X))
        return p[:, 1] if p.ndim == 2 else p
    return np.clip(np.asarray(model.predict(X), dtype=float), 0, 1)


def append_to_sheet(new_rows: pd.DataFrame):
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        existing = conn.read(worksheet="predictions", ttl=0)
        existing = existing.dropna(how="all").dropna(axis=1, how="all") if existing is not None else pd.DataFrame()
    except Exception:
        existing = pd.DataFrame()
    combined = pd.concat([existing, new_rows], ignore_index=True)
    conn.update(worksheet="predictions", data=combined)


# ------------------------------------------------------------------
# 画面メイン描画処理
# ------------------------------------------------------------------
st.subheader("① レース情報")
c1, c2 = st.columns(2)
with c1:
    venue = st.selectbox("競馬場（Venue）", VENUES, index=VENUES.index("東京"))
    track = st.selectbox("馬場状態（Track Condition）", TRACKS)
with c2:
    distance = st.selectbox("距離（Distance）", DISTANCES, index=DISTANCES.index(1600))
    weather = st.selectbox("天気（Weather）", WEATHERS)
race_name = st.text_input("レース名（任意・記録用）", placeholder="例：菊花賞")

st.subheader("② 出馬表を貼り付け")
csv_text = st.text_area(
    "出馬表CSV",
    height=320,
    label_visibility="collapsed",
    placeholder="ここに整形したCSVデータを貼り付けてください",
)

run = st.button("🔮 AI予想を実行する", type="primary", use_container_width=True)

if run:
    if not csv_text.strip():
        st.warning("出馬表CSVを貼り付けてください。")
    else:
        try:
            raw = parse_csv(csv_text)
            df = preprocess(raw, venue, distance, track, weather)
            pace = diagnose_pace(df)

            feats = get_feature_names(model, meta) or DEFAULT_FEATURES
            X, missing = build_X(df, feats)

            # 🚨 【超重要】LightGBMの型エラーを強制回避（文字データを数値ハッシュに高速変換）
            for col in X.columns:
                if col in ["Venue", "Track Condition", "Weather", "Jockey", "Trainer", "Sire", "Sex"]:
                    X[col] = X[col].astype(str).map(lambda v: float(zlib.crc32(v.encode()) % 100))
                else:
                    X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(float)

            # 順番をDEFAULT_FEATURESと完全に一致させて予測を実行
            proba = predict_top3(model, X[feats])

            now = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
            out = pd.DataFrame({
                "馬番": df["Horse Number"].astype(int),
                "馬名": df["Horse Name"],
                "種牡馬": df["Sire"],
                "想定脚質": df["脚質ラベル"],
                "AI予想確率(%)": np.round(np.asarray(proba, dtype=float) * 100, 1),
            })
            out = out.sort_values("AI予想確率(%)", ascending=False, kind="stable").reset_index(drop=True)
            out.insert(0, "順位", np.arange(1, len(out) + 1))

            st.session_state["result"] = out
            st.session_state["pace"] = pace

            # スプレッドシートに蓄積するためのログを作成
            log = out.copy()
            log.insert(0, "実行日時", now)
            log.insert(1, "レース名", race_name)
            log.insert(2, "競馬場", venue)
            log.insert(3, "距離", distance)
            log.insert(4, "馬場状態", track)
            log.insert(5, "天気", weather)
            log.insert(6, "想定ペース", pace.get("pace", "不明"))

            try:
                append_to_sheet(log)
                st.session_state["saved"] = True
            except Exception as e:
                st.session_state["saved"] = False

        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"予想中にエラーが発生しました: {e}")

# 🔮 画面への最終結果出力エリア
if "result" in st.session_state:
    out = st.session_state["result"]
    render_pace(st.session_state.get("pace", {"ok": False}), out)

    st.subheader(f"④ AI予想結果（全{len(out)}頭・3着以内確率順）")
    st.dataframe(
        out,
        hide_index=True,
        use_container_width=True,
        height=int(38 * (len(out) + 1)),
        column_config={
            "順位": st.column_config.NumberColumn("順位", width="small"),
            "馬番": st.column_config.NumberColumn("馬番", width="small"),
            "AI予想確率(%)": st.column_config.ProgressColumn(
                "AI予想確率(%)", format="%.1f%%", min_value=0, max_value=100
            ),
        },
    )
    if st.session_state.get("saved"):
        st.success("データ蓄積完了")
