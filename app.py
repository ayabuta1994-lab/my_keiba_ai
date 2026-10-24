# -*- coding: utf-8 -*-
"""
🏇 競馬予想AI オールインワン（23列仕様 完全対応版）

23列 = 基本21列 + Passing + Race_Key_Src
  Execution Date, Race Day, Race Name, Venue, Distance, Track Condition, Weather, Bracket, Horse Number, Horse Name,
  Sex, Horse Age, Burden Weight, Jockey, Trainer, Tactics, Last 3F, AI_Probability, AI_Rank, Placing, Track_Type, Passing, Race_Key_Src

スプレッドシートのタブ構成
  出走表(ENTRY_SHEET)      : 手入力 / 貼り付け / CSVアップロードの入力バッファ。予想保存後に自動でリセット
  race_results             : 確定結果 + AI予想（23列 + 補助列）
  predictions              : 学習用・履歴特徴量参照用DB（23列）
  race_management          : Race IDをキーに 未予想 → 予想済み → 結果登録済み を一元管理
  horses_master            : 馬名・Sire・Dam's Sire（23列とは独立）
  odds_master              : オッズ専用（ChatGPT等への問い合わせ結果を貼り付け/手入力。学習データには混ぜず、アプリで動的結合）
更新は『Race IDに基づく純粋なUpsert』のみ（上書き前データの退避処理は持たない）。
"""
import io
import pickle
import re
import traceback
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="競馬予想AI オールインワン", page_icon="🐴", layout="wide")

MODEL_PATH = Path(__file__).parent / "model.pkl"
JST = timezone(timedelta(hours=9))


def _st_version():
    m = re.match(r"(\d+)\.(\d+)", str(getattr(st, "__version__", "0.0")))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


STRETCH = {"width": "stretch"} if _st_version() >= (1, 50) else {"use_container_width": True}

# ============================================================
# 定数（23列仕様）
# ============================================================
COLUMNS_21 = ["Execution Date", "Race Day", "Race Name", "Venue", "Distance", "Track Condition", "Weather", "Bracket", "Horse Number",
              "Horse Name", "Sex", "Horse Age", "Burden Weight", "Jockey", "Trainer", "Tactics", "Last 3F", "AI_Probability",
              "AI_Rank", "Placing", "Track_Type"]
COLUMNS_23 = COLUMNS_21 + ["Passing", "Race_Key_Src"]

ENTRY_SHEET = "出走表"
RESULTS_SHEET = "race_results"
PREDICTIONS_SHEET = "predictions"
RACE_MANAGEMENT_SHEET = "race_management"
HORSES_SHEET = "horses_master"
ODDS_SHEET = "odds_master"

RESULT_COLUMNS = COLUMNS_23 + ["Race ID", "Top3 Probability", "Pace", "High Pace Prob", "Front Ratio", "Predicted At", "Updated At"]
MANAGEMENT_COLUMNS = ["Race ID", "Race Day", "Venue", "Race Name", "Distance", "Status", "Horse Count", "Registered At", "Updated At"]
ODDS_COLUMNS = ["Race_Key_Src", "Race Day", "Venue", "Race Name", "Horse Number", "Horse Name", "Win Odds", "Place Odds Min",
                "Place Odds Max", "Popularity", "Odds Time", "Fetched At"]
HORSES_COLUMNS = ["Horse Name", "Sire", "Dam's Sire"]
SHEET_SPECS = {ENTRY_SHEET: COLUMNS_23, PREDICTIONS_SHEET: COLUMNS_23, RESULTS_SHEET: RESULT_COLUMNS,
               RACE_MANAGEMENT_SHEET: MANAGEMENT_COLUMNS, ODDS_SHEET: ODDS_COLUMNS, HORSES_SHEET: HORSES_COLUMNS}
NUM_COLS = {"Distance", "Bracket", "Horse Number", "Horse Age", "Burden Weight", "Last 3F", "AI_Probability", "AI_Rank", "Placing",
            "Track_Type", "Tactics", "Top3 Probability", "High Pace Prob", "Front Ratio", "Horse Count", "Win Odds",
            "Place Odds Min", "Place Odds Max", "Popularity"}
RESULT_FIELDS = ["Placing", "Last 3F", "Passing", "Tactics"]          # 結果で更新される列

STATUS_UNPREDICTED, STATUS_PREDICTED, STATUS_RESULT = "未予想", "予想済み", "結果登録済み"
STATUS_ORDER = {STATUS_UNPREDICTED: 0, STATUS_PREDICTED: 1, STATUS_RESULT: 2}
VERIFY_OPTIONAL_COLS = ()     # 検証時に空欄を許容する列（例: ("Last 3F",)）。既定は23列すべて必須
TACTIC_NAMES = {1: "nige", 2: "senko", 3: "sashi", 4: "oikomi"}
TACTICS_LABEL = {1: "逃げ", 2: "先行", 3: "差し", 4: "追込"}
VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]

INVIS_RE = re.compile(r"[\u200b-\u200f\u2060-\u2064\u202a-\u202e\ufeff\u00a0]")
ALIASES = {
    "着順": "Placing", "確定着順": "Placing", "順位": "Placing", "日付": "Race Day", "開催日": "Race Day", "レース日付": "Race Day",
    "レース名": "Race Name", "競走名": "Race Name", "場名": "Venue", "競馬場": "Venue", "開催": "Venue", "距離": "Distance",
    "馬場": "Track Condition", "馬場状態": "Track Condition", "天気": "Weather", "天候": "Weather", "枠番": "Bracket", "枠": "Bracket",
    "馬番": "Horse Number", "馬名": "Horse Name", "性齢": "SexAge", "性別": "Sex", "年齢": "Horse Age", "斤量": "Burden Weight",
    "負担重量": "Burden Weight", "騎手": "Jockey", "調教師": "Trainer", "厩舎": "Trainer", "脚質": "Tactics", "上がり3F": "Last 3F",
    "上り3F": "Last 3F", "上り": "Last 3F", "後3F": "Last 3F", "芝・ダート": "Track_Type", "芝ダ": "Track_Type",
    "通過": "Passing", "通過順位": "Passing", "通過順": "Passing", "レースID": "Race_Key_Src", "race_id": "Race_Key_Src",
    "父": "Sire", "母父": "Dam's Sire", "母の父": "Dam's Sire",
}
TT_MAP = {"芝": 0, "ダ": 1, "ダート": 1, "障": 2, "障害": 2}


# ============================================================
# 汎用ユーティリティ（train_kaggle.py と同一定義を含む）
# ============================================================
def norm_name(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return ""
    return re.sub(r"[\s\u200b-\u200f\u2060\ufeff]", "", unicodedata.normalize("NFKC", str(v)))


def now_str() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")


def decode_bytes(b: bytes) -> str:
    for enc in ("utf-8-sig", "cp932", "utf-16"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", errors="replace")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def num(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce")


def tactic_num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return np.nan
    s = unicodedata.normalize("NFKC", str(v)).strip()
    m = re.fullmatch(r"([1-4])(?:\.0+)?", s)
    if m: return float(m.group(1))
    for token, n in [("逃", 1), ("先", 2), ("差", 3), ("追", 4)]:
        if token in s: return float(n)
    return np.nan


def parse_passing(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return []
    s = unicodedata.normalize("NFKC", str(v)).strip()
    if "-" not in s and "," in s: s = s.replace(",", "-")
    out = []
    for tok in s.split("-"):
        m = re.search(r"\d+", tok)
        out.append(float(m.group()) if m else np.nan)
    return out


def reference_position(v):
    for p in parse_passing(v)[:2]:
        if pd.notna(p) and p > 0: return p
    return np.nan

def last_corner_position(v):
    """Last recorded corner position; used for prior-run position/recovery features."""
    positions = [p for p in parse_passing(v) if pd.notna(p) and p > 0]
    return positions[-1] if positions else np.nan


def derive_tactics(df):
    """Passing(1〜2角)から v4.2 定量化脚質(1逃げ/2先行/3差し/4追込)。train_kaggle.py と同一ルール。"""
    ref = df["Passing"].map(reference_position)
    n_field = df.groupby("Race_ID")["Horse Name"].transform("size")
    max_pos = ref.groupby(df["Race_ID"]).transform("max").fillna(0)
    n = np.maximum(n_field.values, max_pos.values).astype(float)
    pct = ref.values / np.where(n > 0, n, 1)
    r = ref.values
    code = np.select([np.isnan(r), r <= 1, (r <= 2) | (pct <= 0.30), pct <= 0.70], [np.nan, 1.0, 2.0, 3.0], default=4.0)
    return pd.Series(code, index=df.index)


# ======================================================================
# 共通特徴量コア（train_kaggle.py と app.py で【同一定義】。変更する場合は両ファイルを同時に更新すること）
#   ① 客観的物理データ : 斤量 / 前走着差(※) / 前走上がり3F差 / レース間隔日数
#   ② 馬自身の条件別適性: 距離カテゴリ別・馬場(良/道悪)別・競馬場別
#   ③ 騎手・調教師・血統の条件別適性 / コース別脚質傾向（勝率・複勝率ベース、ベイズ平滑化）
#   ④ クラス変動 / 出走頭数
#   ※ 23列に「着差(タイム差)」列は存在しないため、前走着差は『前走の相対着順 (着順-1)/(頭数-1)』で代替。
#      Margin列(秒差)が将来追加された場合は _pr の算出を差し替えるだけで対応可能。
#   当日馬場バイアス・回収率系の指標は意図的に学習へ入れない（ノイズ・過学習防止）。
# ======================================================================
from collections import defaultdict
FEATURE_VERSION = "v7.2-multisource-history"
SM_A = 5.0
PRIOR_WIN, PRIOR_TOP3, PRIOR_FRONT = 0.07, 0.21, 0.35

HIST_FEATURES = ["Hist_Starts", "Hist_Win_Rate", "Hist_Top3_Rate", "Hist_Mean_Placing3", "Hist_Last3F_Mean3", "Hist_Last3F_Last",
                 "Hist_Last_Tactics", "Rel_Last_Margin", "Rel_Last3F_Diff", "Rel_Last3F_Pct", "Rel_Last_Margin_Mean3", "Rel_Last3F_Diff_Mean3",
                 "Hist_Rest_Days", "Rel_Last_Pos_Recovery"]

COND_FEATURES = ["Hist_Dist_Win", "Hist_Dist_Top3", "Hist_Going_Top3", "Hist_Venue_Top3",
                 "Jockey_VD_Top3", "Trainer_VD_Top3",
                 "Sire_Cond_Top3", "DamSire_Cond_Top3", "Course_Front_Rate", "Class_Level", "Class_Change", "Field_Type", "Dist_Cat"]

CHANGE_FEATURES = ["Diff_Distance", "Diff_Burden_Weight", "Rest_Type", "Rest_Is_Short", "Rest_Is_Long",
                   "Class_Up", "Class_Same", "Class_Down", "Hist_Prev_Tactics", "Tactics_Change", "Tactics_Changed"]
STAMINA_FEATURES = ["Max_Past_Dist", "Dist_Over_Max", "Past_Long_Starts", "Past_Long_Top3_Rate"]
LONG_EXP_DIST = 2400
PREP_RACES = {1: r"セントライト記念|神戸新聞杯|東京優駿|日本ダービー|ラジオNIKKEI賞", 2: r"阪神大賞典|日経賞|大阪杯|ダイヤモンドS|有馬記念", 3: r"天皇賞.{0,2}秋|ジャパンカップ|菊花賞|アルゼンチン共和国杯|ステイヤーズS"}
PREP_RE = {k: re.compile(v) for k, v in PREP_RACES.items()}
PREP_FEATURES = ["Prep_Last_Is_Key", "Prep_Last_Placing", "Prep_Last_Margin", "Prep_Last3F_Diff", "Prep_Key_In_Last3"]
GOING_FEATURES = ["Going_Fit", "Going_Starts", "Is_Heavy_Going", "Going_Fit_Heavy", "Sire_Going_Fit", "DamSire_Going_Fit"]
POS_FEATURES = ["Hist_CornerPos_Last", "Hist_CornerPos_Mean3"]

PEOPLE_STAT_FEATURES = ["Jockey_Top3_All", "Trainer_Top3_All", "Sire_Top3_All", "DamSire_Top3_All"]
BUILDER_FEATURES = HIST_FEATURES + COND_FEATURES + CHANGE_FEATURES + STAMINA_FEATURES + GOING_FEATURES + POS_FEATURES + PEOPLE_STAT_FEATURES

def _sm(n, k, p): return (k + SM_A * p) / (n + SM_A)
def dist_cat(d): return 0 if pd.isna(d) or d < 1400 else (1 if d < 1800 else (2 if d <= 2400 else 3))
def going_class(s):
    """0=good, 1=any wet/heavy going, 2=unknown (not conflated with good)."""
    if pd.isna(s): return 2
    v = unicodedata.normalize("NFKC", str(s)).strip()
    if v in ("良", "0", "0.0"): return 0
    if v in ("稍重", "重", "不良", "稍", "不", "1.0", "2.0", "3.0", "1", "2", "3"): return 1
    return 2
def class_level(name):
    s = unicodedata.normalize("NFKC", str(name)).upper().replace("Ⅲ", "III").replace("Ⅱ", "II").replace("Ⅰ", "I")
    if re.search(r"G\s*(1|I)(?![I0-9])", s): return 8.0
    if re.search(r"G\s*(2|II)(?!I)", s): return 7.0
    if re.search(r"G\s*(3|III)", s): return 6.0
    if re.search(r"オープン|\bOP\b|リステッド|\(L\)|ステークス|特別競走", s) and not re.search(r"[123]勝", s): return 5.0
    if re.search(r"3勝|1600万", s): return 4.0
    if re.search(r"2勝|1000万", s): return 3.0
    if re.search(r"1勝|500万", s): return 2.0
    if "未勝利" in s: return 1.0
    if "新馬" in s: return 0.0
    return np.nan

def prep_columns(df):
    d = df.copy()
    g = d.groupby("Race_ID")
    d["_day"] = pd.to_datetime(d["Race Day"], errors="coerce")
    d["_horse"], d["_venue"] = d["Horse_Key"], d["Venue"].map(norm_name)
    d["_rname"] = d["Race Name"].astype(str).map(lambda x: unicodedata.normalize("NFKC", x))
    d["_rg"] = d["Race Name"].map(race_group_code)
    d["_dist"] = pd.to_numeric(d["Distance"], errors="coerce")
    d["_bw"] = pd.to_numeric(d["Burden Weight"], errors="coerce")
    d["_tt"] = pd.to_numeric(d["Track_Type"], errors="coerce")
    d["_going"] = d["Track Condition"].map(going_class)
    d["_dc"] = d["_dist"].map(dist_cat)
    cs = d["Class_Src"].fillna("").astype(str) if "Class_Src" in d.columns else pd.Series("", index=d.index)
    d["_cls"] = pd.Series(np.where(cs.str.strip() != "", d["Race Name"].astype(str) + " " + cs, d["Race Name"].astype(str)), index=d.index).map(class_level)
    d["_field"] = g["Horse Name"].transform("size").astype(float)
    d["_pl"] = pd.to_numeric(d["Placing_num"], errors="coerce")
    d["_l3f"] = pd.to_numeric(d["Last 3F"], errors="coerce")
    d["_l3d"] = d["_l3f"] - d.groupby("Race_ID")["_l3f"].transform("min")
    d["_pr"] = (d["_pl"] - 1) / np.maximum(d["_field"] - 1, 1)
    if "Passing" in d.columns:
        # Use the last recorded corner (usually 4C on four-corner courses), not
        # the first two corner positions used by the running-style classifier.
        ref = d["Passing"].map(last_corner_position)
        n_eff = np.maximum(d["_field"], ref.groupby(d["Race_ID"]).transform("max").fillna(0))
        d["_cpf"] = ref / n_eff.where(n_eff > 0)
    else: d["_cpf"] = np.nan
    d["_l3p"] = (d.groupby("Race_ID")["_l3f"].rank(method="min", ascending=True) - 1) / np.maximum(d.groupby("Race_ID")["_l3f"].transform("count") - 1, 1)
    d["_tac"] = pd.to_numeric(d["Tactics"], errors="coerce")
    d["_jk"], d["_tr"] = d["Jockey"].map(norm_name), d["Trainer"].map(norm_name)
    d["_sire"], d["_dam"] = d["Sire"].fillna("Unknown").astype(str), d["Dam's Sire"].fillna("Unknown").astype(str)
    return d

INTERNAL_COLS = ["_day", "_horse", "_venue", "_rname", "_rg", "_dist", "_bw", "_tt", "_going", "_dc", "_cls", "_field", "_pl", "_l3f", "_l3d", "_l3p", "_pr", "_cpf", "_tac", "_jk", "_tr", "_sire", "_dam"]

class FeatureBuilder:
    def __init__(self):
        self.h = defaultdict(list); self.c = {}
    def _bump(self, key, w, t):
        v = self.c.get(key)
        if v is None: v = self.c[key] = [0, 0, 0]
        v[0] += 1; v[1] += w; v[2] += t
    @staticmethod
    def _keys(r):
        dc, gc, vn = r["_dc"], r["_going"], r["_venue"]
        return [("hd", r["_horse"], dc), ("hg", r["_horse"], gc), ("hv", r["_horse"], vn),
                ("jvd", r["_jk"], vn, dc), ("tvd", r["_tr"], vn, dc),
                ("sc", r["_sire"], vn, dc, gc), ("dm", r["_dam"], vn, dc, gc),
                ("sg", r["_sire"], gc), ("dg", r["_dam"], gc), ("sa", r["_sire"]), ("da", r["_dam"]),
                ("ja", r["_jk"]), ("ta", r["_tr"])]
    def add(self, r):
        pl = r["_pl"]
        if pd.isna(pl): return
        w, t = int(pl == 1), int(pl <= 3)
        for k in self._keys(r):
            if k[1] in ("", "Unknown"): continue
            self._bump(k, w, t)
        if pd.notna(r["_tac"]): self._bump(("course", r["_venue"], r["_dist"], r["_tt"]), int(r["_tac"] in (1, 2)), 0)
        self.h[r["_horse"]].append(r)
    def _rate(self, key, col, prior):
        v = self.c.get(key); return _sm(v[0], v[col], prior) if v else prior
    def _balance_features(self, r, past):
        hs, gc = r["_horse"], r["_going"]
        n = len(past); k3 = sum(1 for p in past if p["_pl"] <= 3)
        overall = _sm(n, k3, PRIOR_TOP3)
        gfit = self._rate(("hg", hs, gc), 2, PRIOR_TOP3) - overall
        v = self.c.get(("hg", hs, gc))
        cps = [p.get("_cpf", np.nan) for p in past[-3:] if pd.notna(p.get("_cpf", np.nan))]
        lastcp = past[-1].get("_cpf", np.nan) if n and pd.notna(past[-1].get("_cpf", np.nan)) else None
        sire, dam, jk, tr = r["_sire"], r["_dam"], r["_jk"], r["_tr"]
        return {
            "Going_Fit": float(gfit), "Going_Starts": float(v[0]) if v else 0.0, "Is_Heavy_Going": float(gc),
            "Going_Fit_Heavy": float(gfit) if gc == 1 else 0.0,
            "Sire_Going_Fit": self._rate(("sg", sire, gc), 2, PRIOR_TOP3) - self._rate(("sa", sire), 2, PRIOR_TOP3),
            "DamSire_Going_Fit": self._rate(("dg", dam, gc), 2, PRIOR_TOP3) - self._rate(("da", dam), 2, PRIOR_TOP3),
            "Hist_CornerPos_Last": float(lastcp) if lastcp is not None else -1.0,
            "Hist_CornerPos_Mean3": float(np.mean(cps)) if cps else -1.0,
            "Jockey_Top3_All": self._rate(("ja", jk), 2, PRIOR_TOP3),
            "Trainer_Top3_All": self._rate(("ta", tr), 2, PRIOR_TOP3),
            "Sire_Top3_All": self._rate(("sa", sire), 2, PRIOR_TOP3), "DamSire_Top3_All": self._rate(("da", dam), 2, PRIOR_TOP3),
        }

    @staticmethod
    def _prep_features(r, past):
        out = {"Prep_Last_Is_Key": 0.0, "Prep_Last_Placing": 0.0, "Prep_Last_Margin": -1.0, "Prep_Last3F_Diff": -1.0, "Prep_Key_In_Last3": 0.0}
        g = int(r.get("_rg", 0) or 0)
        if g == 0 or not past: return out
        pat = PREP_RE[g]; out["Prep_Key_In_Last3"] = float(any(pat.search(p["_rname"]) for p in past[-3:]))
        last = past[-1]
        if pat.search(last["_rname"]):
            out.update({"Prep_Last_Is_Key": 1.0, "Prep_Last_Placing": float(last["_pl"]), "Prep_Last_Margin": float(last["_pr"]) if pd.notna(last["_pr"]) else -1.0, "Prep_Last3F_Diff": float(last["_l3d"]) if pd.notna(last["_l3d"]) else -1.0})
        return out

    @staticmethod
    def _stamina_features(r, past):
        dists = [(p["_dist"], p["_pl"]) for p in past if pd.notna(p["_dist"])]
        cur, mx = r["_dist"], max((d for d, _ in dists), default=None)
        longs = [pl for d, pl in dists if d >= LONG_EXP_DIST]
        return {"Max_Past_Dist": float(mx) if mx is not None else 0.0, "Dist_Over_Max": float(cur - mx) if (mx is not None and pd.notna(cur)) else 0.0, "Past_Long_Starts": float(len(longs)), "Past_Long_Top3_Rate": _sm(len(longs), sum(1 for x in longs if x <= 3), PRIOR_TOP3)}

    @staticmethod
    def _change_features(r, past, tac, last):
        def diff(cur, prv): return float(cur - prv) if (pd.notna(cur) and pd.notna(prv)) else 0.0
        days = (r["_day"] - last["_day"]).days if (last is not None and pd.notna(r["_day"]) and pd.notna(last["_day"])) else None
        rtype = 0.0 if days is None else (1.0 if days <= 21 else (2.0 if days <= 42 else (3.0 if days <= 89 else (4.0 if days <= 179 else 5.0))))
        cur, prv = r["_cls"], (last["_cls"] if last is not None else np.nan)
        known = pd.notna(cur) and pd.notna(prv)
        last_t, prev_t = tac[-1] if len(tac) >= 1 else None, tac[-2] if len(tac) >= 2 else None
        both = last_t is not None and prev_t is not None
        return {"Diff_Distance": diff(r["_dist"], last["_dist"]) if last is not None else 0.0, "Diff_Burden_Weight": diff(r["_bw"], last["_bw"]) if last is not None else 0.0,
                "Rest_Type": rtype, "Rest_Is_Short": 1.0 if rtype == 1.0 else 0.0, "Rest_Is_Long": 1.0 if rtype >= 4.0 else 0.0,
                "Class_Up": 1.0 if known and cur > prv else 0.0, "Class_Same": 1.0 if known and cur == prv else 0.0, "Class_Down": 1.0 if known and cur < prv else 0.0,
                "Hist_Prev_Tactics": float(prev_t) if prev_t is not None else 0.0, "Tactics_Change": float(last_t - prev_t) if both else 0.0, "Tactics_Changed": 1.0 if both and last_t != prev_t else 0.0}

    def features(self, r):
        past = self.h.get(r["_horse"], []); n = len(past)
        pls = [p["_pl"] for p in past]; l3 = past[-3:]
        l3f3 = [p["_l3f"] for p in l3 if pd.notna(p["_l3f"])]
        l3all = [p["_l3f"] for p in past if pd.notna(p["_l3f"])]
        tac = [p["_tac"] for p in past if pd.notna(p["_tac"])]
        last = past[-1] if n else None
        vn, dc, gc = r["_venue"], r["_dc"], r["_going"]
        cl = r["_cls"]; fs = r.get("_field", np.nan)
        pr3 = [p["_pr"] for p in l3 if pd.notna(p["_pr"])]; l3d3 = [p["_l3d"] for p in l3 if pd.notna(p["_l3d"])]
        
        pos_recovery = 0.0
        if last is not None and pd.notna(last.get("_cpf")) and pd.notna(last.get("_pl")) and pd.notna(last.get("_field")):
            last_corner_pos = last["_cpf"] * last["_field"]
            pos_recovery = float(last_corner_pos - last["_pl"])

        f = {
            "Hist_Starts": float(n), "Hist_Win_Rate": float(np.mean([x == 1 for x in pls])) if n else 0.0, "Hist_Top3_Rate": float(np.mean([x <= 3 for x in pls])) if n else 0.0,
            "Hist_Mean_Placing3": float(np.mean([p["_pl"] for p in l3])) if l3 else 0.0, "Hist_Last3F_Mean3": float(np.mean(l3f3)) if l3f3 else 0.0,
            "Hist_Last3F_Last": float(l3all[-1]) if l3all else 0.0, "Hist_Last_Tactics": float(tac[-1]) if tac else 0.0,
            "Rel_Last_Margin": float(last["_pr"]) if last is not None and pd.notna(last["_pr"]) else -1.0,
            "Rel_Last3F_Diff": float(last["_l3d"]) if last is not None and pd.notna(last["_l3d"]) else -1.0,
            "Rel_Last3F_Pct": float(last["_l3p"]) if last is not None and pd.notna(last["_l3p"]) else -1.0,
            "Rel_Last_Margin_Mean3": float(np.mean(pr3)) if pr3 else -1.0, "Rel_Last3F_Diff_Mean3": float(np.mean(l3d3)) if l3d3 else -1.0,
            "Hist_Rest_Days": float(min((r["_day"] - last["_day"]).days, 365)) if last is not None and pd.notna(r["_day"]) and pd.notna(last["_day"]) else -1.0,
            "Rel_Last_Pos_Recovery": pos_recovery,
            "Hist_Dist_Win": self._rate(("hd", r["_horse"], dc), 1, PRIOR_WIN), "Hist_Dist_Top3": self._rate(("hd", r["_horse"], dc), 2, PRIOR_TOP3),
            "Hist_Going_Top3": self._rate(("hg", r["_horse"], gc), 2, PRIOR_TOP3), "Hist_Venue_Top3": self._rate(("hv", r["_horse"], vn), 2, PRIOR_TOP3),
            "Jockey_VD_Top3": self._rate(("jvd", r["_jk"], vn, dc), 2, PRIOR_TOP3),
            "Trainer_VD_Top3": self._rate(("tvd", r["_tr"], vn, dc), 2, PRIOR_TOP3),
            "Sire_Cond_Top3": self._rate(("sc", r["_sire"], vn, dc, gc), 2, PRIOR_TOP3), "DamSire_Cond_Top3": self._rate(("dm", r["_dam"], vn, dc, gc), 2, PRIOR_TOP3),
            "Course_Front_Rate": self._rate(("course", vn, r["_dist"], r["_tt"]), 1, PRIOR_FRONT),
            "Class_Level": float(cl) if pd.notna(cl) else -1.0, "Class_Change": float(cl - last["_cls"]) if pd.notna(cl) and last is not None and pd.notna(last["_cls"]) else 0.0,
            "Dist_Cat": float(dc), "Field_Type": 0.0 if (pd.notna(fs) and fs <= 9) else (2.0 if (pd.notna(fs) and fs >= 16) else 1.0),
        }
        f.update(self._change_features(r, past, tac, last))
        f.update(self._stamina_features(r, past))
        f.update(self._prep_features(r, past))
        f.update(self._balance_features(r, past))
        return f

def add_pace_features(df, tac_col="Hist_Last_Tactics", race_col="Race_ID"):
    """各馬の『前走脚質』をレース単位で集計（予想時にも取得できる情報のみ）。"""
    t = pd.to_numeric(df[tac_col], errors="coerce").fillna(0)
    for code, name in TACTIC_NAMES.items():
        df[f"exp_count_{name}"] = (t == code).astype(int).groupby(df[race_col]).transform("sum").astype(float)
    known = t.between(1, 4).astype(int).groupby(df[race_col]).transform("sum")
    front = t.isin([1, 2]).astype(int).groupby(df[race_col]).transform("sum")
    df["exp_front_ratio"] = np.where(known > 0, front / known.replace(0, 1), 0.0)
    df["field_size"] = df.groupby(race_col)["Horse Name"].transform("size").astype(float)
    return df

# ⑥ メンバー内での相対評価：同一レースの出走馬どうしを比較（出走表が確定すれば予想時にも同じ値が作れる＝リークなし）
#   *_Rel : メンバー平均との差 / *_Pct : メンバー内の相対順位（0に近いほど優秀、0.5=データなし）
MEMBER_FEATURES = ["Mem_Last3F_Diff_Pct", "Mem_Last3F_Mean3_Rel", "Mem_Last_Margin_Rel", "Mem_Last_Margin_Pct",
                   "Mem_Win_Rate_Rel", "Mem_Top3_Rate_Rel", "Mem_Top3_Rate_Pct", "Mem_Placing3_Rel", "Mem_Burden_Rel", "Mem_Dist_Top3_Rel",
                   # 展開・馬場の適性（レース内で全馬共通の値を、馬ごとの有利不利に変換）
                   "Pace_Pressure", "Pace_Advantage", "Mem_Front_Rank", "Nige_Rivals", "Front_Rivals", "Going_Fit_Rel"]

def add_member_relative_features(df, race_col="Race_ID"):
    def col(name): return pd.to_numeric(df[name], errors="coerce")
    started = col("Hist_Starts").fillna(0) > 0                     # 過去走のある馬だけで比較（初出走は中立値）
    def rel(out, name, valid, better="low", pct=False):
        x = col(name).where(valid); g = x.groupby(df[race_col])
        if pct: df[out] = g.rank(method="average", ascending=(better == "low"), pct=True).fillna(0.5).astype(float)
        else: df[out] = (x - g.transform("mean")).fillna(0.0).astype(float)
    rel("Mem_Last3F_Diff_Pct", "Rel_Last3F_Diff", col("Rel_Last3F_Diff") >= 0, "low", pct=True)
    rel("Mem_Last3F_Mean3_Rel", "Hist_Last3F_Mean3", col("Hist_Last3F_Mean3") > 0)
    rel("Mem_Last_Margin_Rel", "Rel_Last_Margin", col("Rel_Last_Margin") >= 0)
    rel("Mem_Last_Margin_Pct", "Rel_Last_Margin", col("Rel_Last_Margin") >= 0, "low", pct=True)
    rel("Mem_Win_Rate_Rel", "Hist_Win_Rate", started)
    rel("Mem_Top3_Rate_Rel", "Hist_Top3_Rate", started)
    rel("Mem_Top3_Rate_Pct", "Hist_Top3_Rate", started, "high", pct=True)
    rel("Mem_Placing3_Rel", "Hist_Mean_Placing3", started)
    rel("Mem_Burden_Rel", "Burden Weight", col("Burden Weight").notna())
    rel("Mem_Dist_Top3_Rel", "Hist_Dist_Top3", started)
    rel("Going_Fit_Rel", "Going_Fit", started)                         # 今回の馬場への適性を、同じレースの他馬と比較
    g = df[race_col]
    cpv = col("Hist_CornerPos_Mean3").where(col("Hist_CornerPos_Mean3") >= 0)          # 序盤の位置取り(0=先頭〜1=後方)。判明している馬のみ
    front = (cpv <= 0.25).astype(float).where(cpv.notna())
    df["Pace_Pressure"] = front.groupby(g).transform("mean").fillna(0.3).astype(float)     # 場の『前に行きたい馬』の割合 = 想定ペースの速さ
    df["Pace_Advantage"] = ((cpv - 0.5) * (df["Pace_Pressure"] - 0.30)).fillna(0.0).astype(float)   # +: 速い流れで後ろ寄り／遅い流れで前寄り＝展開が向く
    df["Mem_Front_Rank"] = cpv.groupby(g).rank(method="average", pct=True).fillna(0.5).astype(float)   # 0に近いほどメンバー内で前に行く
    nige = (col("Hist_Last_Tactics") == 1).astype(float)
    df["Nige_Rivals"] = (nige.groupby(g).transform("sum") - nige).astype(float)               # 同じレースの他の逃げ馬の数（競り合い）
    fr = front.fillna(0.0)
    df["Front_Rivals"] = (fr.groupby(g).transform("sum") - fr).astype(float)                  # 同じレースの他の先行馬の数
    return df

# ⑧ 長距離の特別なレース（長距離専用モデルの対象）：菊花賞(3000m) / 天皇賞(春)(3200m) / 有馬記念(2500m)
SPECIAL_RACES = {"Is_Kikkasho": r"菊花賞", "Is_TennoSho_Spring": r"天皇賞.{0,2}春", "Is_Arima": r"有馬記念"}
SPECIAL_RACE_FEATURES = list(SPECIAL_RACES)
SPECIAL_RACE_LABELS = {1: "菊花賞", 2: "天皇賞(春)", 3: "有馬記念"}

def race_group_code(name):
    s_ = unicodedata.normalize("NFKC", str(name))
    for i, pat in enumerate(SPECIAL_RACES.values(), 1):
        if re.search(pat, s_): return i
    return 0

def add_race_group_features(df):
    nm = df["Race Name"].astype(str).map(lambda x: unicodedata.normalize("NFKC", x))
    for k, pat in SPECIAL_RACES.items(): df[k] = nm.str.contains(pat, regex=True).astype(float)
    return df
# ======================= 共通特徴量コア ここまで =======================


# ============================================================
# 23列への標準化（standardize_to_23cols）
# ============================================================
def _col(df, c):
    return df[c].fillna("").astype(str) if c in df.columns else pd.Series("", index=df.index)


def race_ids_vec(df: pd.DataFrame) -> pd.Series:
    """Race ID = Race_Key_Src。無い行のみ『日付_会場_レース名_距離』で代替（ユニークキー）。"""
    src = _col(df, "Race_Key_Src").str.replace(r"\.0$", "", regex=True).str.strip()
    dist = _col(df, "Distance").str.replace(r"\.0$", "", regex=True)
    comp = _col(df, "Race Day").str[:10] + "_" + _col(df, "Venue") + "_" + _col(df, "Race Name") + "_" + dist
    return src.where(src != "", comp)


def standardize_to_23cols(df: pd.DataFrame, defaults: dict = None, extra_cols=()) -> pd.DataFrame:
    """任意の出走表/結果/CSVを23列へ展開・補完する。
    ・別名列(性齢/馬名/馬番…)を正規名へ、「牡3」→ Sex:牡 / Horse Age:3 に自動分解
    ・欠損列は空欄で補完（旧仕様の『21列固定で空欄を作る』手作業は不要）
    ・'**' 等のマスク値は空欄扱い。数値列は数値化。Race_Key_Src は文字列（.0除去）"""
    defaults = defaults or {}
    d = df.copy()
    d.columns = [INVIS_RE.sub("", str(c)).strip() for c in d.columns]
    d = d.rename(columns={c: ALIASES[c] for c in d.columns if c in ALIASES and ALIASES[c] not in d.columns})
    d = d.loc[:, ~d.columns.duplicated()]
    d = d.astype(object).where(d.notna(), "").astype(str)
    for c in d.columns:
        d[c] = d[c].str.replace(INVIS_RE, "", regex=True).str.strip()
    d = d.replace({"nan": "", "None": "", "NaT": "", "**": "", "－": ""})
    if "SexAge" in d.columns:
        ex = d["SexAge"].str.extract(r"([牡牝セ騸])\s*(\d+)")
        for tgt, src in (("Sex", ex[0]), ("Horse Age", ex[1])):
            d[tgt] = d[tgt].where(d[tgt] != "", src.fillna("")) if tgt in d.columns else src.fillna("")
    for c in COLUMNS_23:
        if c not in d.columns: d[c] = ""
        if defaults.get(c) not in (None, ""): d[c] = d[c].where(d[c] != "", str(defaults[c]))
    d["Sex"] = d["Sex"].replace({"セ": "セン", "騸": "セン"})
    dt = pd.to_datetime(d["Race Day"], errors="coerce")
    d["Race Day"] = np.where(dt.notna(), dt.dt.strftime("%Y-%m-%d"), d["Race Day"])
    d["Execution Date"] = d["Execution Date"].where(d["Execution Date"] != "", now_str())
    d["Race_Key_Src"] = d["Race_Key_Src"].str.replace(r"\.0$", "", regex=True)
    tt_num = pd.to_numeric(d["Track_Type"], errors="coerce")
    d["Track_Type"] = tt_num.where(tt_num.notna(), d["Track_Type"].map(lambda v: TT_MAP.get(v, np.nan)))
    d["Last 3F"] = clean_last3f_series(d["Last 3F"])
    for c in ["Distance", "Bracket", "Horse Number", "Horse Age", "Burden Weight", "AI_Probability", "AI_Rank", "Placing"]:
        d[c] = pd.to_numeric(d[c].str.replace(",", "", regex=False), errors="coerce")
    d["Tactics"] = d["Tactics"].map(tactic_num)
    keep = COLUMNS_23 + [c for c in extra_cols if c in d.columns]
    return d[keep].reset_index(drop=True)


def clean_last3f_series(s: pd.Series) -> pd.Series:
    """上がり3Fのクレンジング: '**'・空白・記号混じりから『33.5』形式(小数第1位)のみ抽出。範囲外・数値化不能は空欄(NaN)。"""
    t = s.astype(str).map(lambda v: unicodedata.normalize("NFKC", v))
    v = pd.to_numeric(t.str.extract(r"(?<![\d:.])(\d{2}\.\d)(?![\d:])")[0], errors="coerce")
    return v.where(v.between(25.0, 70.0))


def parse_input_table(raw_text: str) -> pd.DataFrame:
    """貼り付け（TSV/CSV）→ DataFrame"""
    sep = "\t" if "\t" in raw_text else ","
    return pd.read_csv(io.StringIO(raw_text), sep=sep, dtype=str, keep_default_na=False)


# ============================================================
# Google スプレッドシート接続・読込・Upsert（バックアップ処理なし）
# ============================================================
@st.cache_resource
def get_spreadsheet():
    creds_dict = None
    for key in ["connections.gsheets", "gcp_service_account", "gsheets"]:
        if key in st.secrets:
            creds_dict = dict(st.secrets[key]); break
    if creds_dict is None and hasattr(st.secrets, "connections") and "gsheets" in st.secrets.connections:
        creds_dict = dict(st.secrets.connections.gsheets)
    if creds_dict is None:
        raise KeyError("Streamlit Secrets にサービスアカウント情報が見つかりません。")
    creds = Credentials.from_service_account_info(creds_dict, scopes=["https://www.googleapis.com/auth/spreadsheets",
                                                                      "https://www.googleapis.com/auth/drive"])
    gc = gspread.authorize(creds)
    ref = creds_dict.get("spreadsheet") or st.secrets.get("spreadsheet_id")
    if not ref: raise ValueError("Secrets に spreadsheet のURLまたはIDが設定されていません。")
    return gc.open_by_url(ref) if "spreadsheets/d/" in str(ref) else gc.open_by_key(ref)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_values(spreadsheet_id, title):
    return get_spreadsheet().worksheet(title).get_all_values()


def clear_sheet_cache():
    _cached_values.clear()


def values_to_dataframe(values) -> pd.DataFrame:
    if not values or len(values) < 2: return pd.DataFrame()
    headers = [str(v).strip() for v in values[0]]
    while headers and not headers[-1]: headers.pop()
    if not headers: return pd.DataFrame()
    n = len(headers)
    rows = [(list(r[:n]) + [""] * (n - len(r))) for r in values[1:] if any(str(x).strip() for x in r)]
    return pd.DataFrame(rows, columns=headers)


def read_sheet_df(sh, title: str, fresh: bool = False) -> pd.DataFrame:
    try:
        if fresh:
            clear_sheet_cache()
            return values_to_dataframe(sh.worksheet(title).get_all_values())
        return values_to_dataframe(_cached_values(sh.id, title))
    except gspread.exceptions.WorksheetNotFound:
        return pd.DataFrame()


def _col_letter(n: int) -> str:
    return gspread.utils.rowcol_to_a1(1, n).rstrip("0123456789")


def ensure_sheet(sh, title, cols, rows=2000):
    """シートが無ければ作成してヘッダーを書き、不足列は右端へ追加。戻り値: (ws, headers)"""
    try:
        ws = sh.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=rows, cols=max(len(cols), 26))
    headers = [h.strip() for h in ws.row_values(1)]
    while headers and not headers[-1]: headers.pop()
    if not headers:
        ws.update(values=[cols], range_name="A1", value_input_option="RAW"); return ws, list(cols)
    missing = [c for c in cols if c not in headers]
    if missing:
        if ws.col_count < len(headers) + len(missing): ws.add_cols(len(headers) + len(missing) - ws.col_count)
        ws.update(values=[missing], range_name=f"{_col_letter(len(headers) + 1)}1", value_input_option="RAW")
        headers += missing
    return ws, headers


def _cell(col, v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return ""
    if col in NUM_COLS:
        try:
            f = float(v); return int(f) if f == int(f) else round(f, 4)
        except Exception:
            return ""
    s = str(v)
    return "" if s in ("nan", "None", "NaT") else s


def df_to_rows(df: pd.DataFrame, headers):
    return [[_cell(h, r.get(h, "")) for h in headers] for r in df.to_dict("records")]


def _append_chunks(ws, rows, chunk=4000):
    for i in range(0, len(rows), chunk):
        ws.append_rows(rows[i:i + chunk], value_input_option="RAW")


def _delete_rows(ws, row_numbers):
    """1始まりの行番号を連続範囲にまとめ、下から一括削除。"""
    nums = sorted(set(row_numbers), reverse=True)
    if not nums: return 0
    ranges, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev - 1: prev = n; continue
        ranges.append((prev, start)); start = prev = n
    ranges.append((prev, start))
    reqs = [{"deleteDimension": {"range": {"sheetId": ws.id, "dimension": "ROWS", "startIndex": s - 1, "endIndex": e}}} for s, e in ranges]
    ws.spreadsheet.batch_update({"requests": reqs})
    return len(nums)


def delete_races(sh, title, race_ids):
    """指定Race IDの行だけを削除（出走表バッファの部分リセット用）。"""
    try:
        ws = sh.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        return 0
    vals = ws.get_all_values()
    body = values_to_dataframe(vals)
    if body.empty: return 0
    ids = race_ids_vec(body)
    n = _delete_rows(ws, [i + 2 for i, x in enumerate(ids) if x in set(race_ids)])
    clear_sheet_cache(); return n


def upsert_by_race(sh, title, df: pd.DataFrame, cols, also_delete=()):
    """【純粋なUpsert】Race IDが一致する既存行を削除 → 新しい行を追加。退避・バックアップ処理は行わない。"""
    ws, headers = ensure_sheet(sh, title, cols)
    new_ids = set(race_ids_vec(df)) | set(also_delete)
    deleted = 0
    body = values_to_dataframe(ws.get_all_values())
    if not body.empty:
        ids = race_ids_vec(body)
        deleted = _delete_rows(ws, [i + 2 for i, x in enumerate(ids) if x in new_ids])
    _append_chunks(ws, df_to_rows(df, headers))
    clear_sheet_cache()
    return {"deleted": deleted, "added": len(df)}


def upsert_horses(sh, df: pd.DataFrame):
    ws, headers = ensure_sheet(sh, HORSES_SHEET, HORSES_COLUMNS)
    cur = values_to_dataframe(ws.get_all_values())
    cur = cur.reindex(columns=HORSES_COLUMNS).fillna("") if not cur.empty else pd.DataFrame(columns=HORSES_COLUMNS)
    new = df.reindex(columns=HORSES_COLUMNS).fillna("")
    allh = pd.concat([cur, new], ignore_index=True)
    allh["_k"] = allh["Horse Name"].map(norm_name)
    allh = allh[allh["_k"] != ""].replace("", np.nan)
    allh = allh.groupby("_k", sort=False, as_index=False).last().fillna("").drop(columns="_k")   # 後勝ち・空欄は前の値で補完
    ws.clear()
    ws.update(values=[HORSES_COLUMNS], range_name="A1", value_input_option="RAW")
    _append_chunks(ws, allh[HORSES_COLUMNS].astype(str).values.tolist())
    clear_sheet_cache()
    return {"total": len(allh), "input": len(new)}


def upload_csv_to_spreadsheet(csv_path, sheet_title, mode="replace", standardize=None, sh=None):
    """【方法B】CSV（パス / bytes / アップロードファイル）を指定タブへ一括同期。
    mode: 'replace'=タブ全体を上書き / 'append'=末尾に追記 / 'upsert'=Race ID単位で更新・追加（horses_master は馬名単位）
    standardize: None=23列系シート(出走表/predictions/race_results)のみ自動で23列標準化"""
    sh = sh or get_spreadsheet()
    if isinstance(csv_path, (str, Path)): raw = Path(csv_path).read_bytes()
    elif hasattr(csv_path, "getvalue"): raw = csv_path.getvalue()
    elif hasattr(csv_path, "read"): raw = csv_path.read()
    else: raw = bytes(csv_path)
    df = pd.read_csv(io.StringIO(decode_bytes(raw)), dtype=str, keep_default_na=False)
    if standardize is None: standardize = sheet_title in (ENTRY_SHEET, PREDICTIONS_SHEET, RESULTS_SHEET)
    cols = SHEET_SPECS.get(sheet_title) or list(df.columns)
    if standardize:
        df = standardize_to_23cols(df, extra_cols=[c for c in cols if c not in COLUMNS_23])
    elif sheet_title == HORSES_SHEET:
        df = df.rename(columns={c: ALIASES.get(c.strip(), c.strip()) for c in df.columns})
    if sheet_title == HORSES_SHEET and mode == "upsert":
        return upsert_horses(sh, df)
    if mode == "upsert":
        r = upsert_by_race(sh, sheet_title, df, cols); r["rows"] = len(df); return r
    ws, headers = ensure_sheet(sh, sheet_title, cols)
    if mode == "replace":
        ws.clear()
        ws.update(values=[headers], range_name="A1", value_input_option="RAW")
    _append_chunks(ws, df_to_rows(df, headers))
    clear_sheet_cache()
    return {"rows": len(df), "mode": mode, "sheet": sheet_title}


# ============================================================
# race_management（Race IDユニーク・ステータス一元管理）
# ============================================================
def load_race_management_df(sh, fresh=True) -> pd.DataFrame:
    d = read_sheet_df(sh, RACE_MANAGEMENT_SHEET, fresh=fresh)
    return d.reindex(columns=MANAGEMENT_COLUMNS).fillna("") if not d.empty else pd.DataFrame(columns=MANAGEMENT_COLUMNS)


def upsert_management(sh, items):
    """items: [{rid, day, venue, name, distance, status, count}]。ステータスは降格させない（結果登録済みは維持）。"""
    ws, headers = ensure_sheet(sh, RACE_MANAGEMENT_SHEET, MANAGEMENT_COLUMNS)
    vals = ws.get_all_values()
    pos = {}
    for i, r in enumerate(vals[1:], start=2):
        if r and r[0].strip(): pos[r[0].strip()] = (i, r)
    ts, adds, updates = now_str(), [], []
    for it in items:
        rid = str(it["rid"])
        if rid in pos:
            i, r = pos[rid]
            r = list(r) + [""] * (len(headers) - len(r))
            row = dict(zip(headers, r))
            old = row.get("Status", "") or STATUS_UNPREDICTED
            new = it["status"] if STATUS_ORDER.get(it["status"], 0) >= STATUS_ORDER.get(old, 0) else old
            row.update({"Status": new, "Updated At": ts})
            if it.get("count"): row["Horse Count"] = it["count"]
            updates.append({"range": f"A{i}:{_col_letter(len(headers))}{i}", "values": [[_cell(h, row.get(h, "")) for h in headers]]})
        else:
            row = {"Race ID": rid, "Race Day": it.get("day", ""), "Venue": it.get("venue", ""), "Race Name": it.get("name", ""),
                   "Distance": it.get("distance", ""), "Status": it["status"], "Horse Count": it.get("count", ""),
                   "Registered At": ts, "Updated At": ts}
            adds.append([_cell(h, row.get(h, "")) for h in headers])
    if updates: ws.batch_update(updates, value_input_option="RAW")
    if adds: ws.append_rows(adds, value_input_option="RAW")
    clear_sheet_cache()


def race_items_from_df(df: pd.DataFrame, status: str):
    out = []
    rids = race_ids_vec(df)
    for rid, g in df.groupby(rids, sort=False):
        r = g.iloc[0]
        out.append({"rid": rid, "day": r["Race Day"], "venue": r["Venue"], "name": r["Race Name"],
                    "distance": r["Distance"], "status": status, "count": len(g)})
    return out


# ============================================================
# 手入力・CSV運用の補助
#   ・オッズ: ChatGPT等のAIへ問い合わせた結果を貼り付け / 手入力 → odds_master
#   ・確定結果: scrape_netkeiba.py のCSV or 手入力 → race_results / predictions へ反映（AI予想列は保持）
# ============================================================
def build_odds_prompt(day, label, race: pd.DataFrame) -> str:
    horses = " / ".join(f"{int(n)}番 {h}" for n, h in zip(race["Horse Number"], race["Horse Name"]) if pd.notna(n))
    return (f"以下のJRAレースの最新の単勝オッズと複勝オッズ（下限・上限）を、JRA公式またはnetkeibaなど信頼できる出典で確認して教えてください。\n"
            f"レース: {day} {label}\n出走馬: {horses}\n\n"
            "出力形式（CSVのみ・ヘッダーなし・説明文なし。不明な値は空欄）:\n馬番,単勝オッズ,複勝オッズ下限,複勝オッズ上限\n"
            "※ 推測や記憶による数値は書かず、確認できた最新の数値だけを出力してください。")


def _first_num(x):
    m = re.search(r"\d+(?:\.\d+)?", x or "")
    return float(m.group()) if m else np.nan


def parse_odds_text(text: str) -> pd.DataFrame:
    """AIの回答（CSV / TSV / 表 / 『1番 3.4倍 (1.5-2.1)』形式）から 馬番・単勝・複勝下限・複勝上限 を抽出。"""
    rows = []
    for line in unicodedata.normalize("NFKC", text or "").splitlines():
        if re.match(r"^\s*\|?\s*\d+\s*[,|]", line) and ("," in line or "|" in line):
            fields = [f.strip() for f in re.split(r"[,|]", line.strip().strip("|"))]
            vals = []
            for f in fields:                                   # 馬名など数字の無い欄は捨て、空欄は欠損として保持、範囲表記は2値に展開
                if f == "": vals.append(np.nan)
                elif re.search(r"\d", f): vals.extend(float(x) for x in re.findall(r"\d+(?:\.\d+)?", f))
            if not vals: continue
            uma, rest = vals[0], vals[1:4]
        else:
            line = re.sub(r"(?<=\d)\s*[-〜~～]\s*(?=\d)", " ", re.sub(r"[;\t、]", " ", line))
            nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", line)]
            if len(nums) < 2: continue
            uma, rest = nums[0], nums[1:4]
        if pd.isna(uma) or uma != int(uma) or not (1 <= uma <= 18): continue
        rest = (rest + [np.nan] * 3)[:3]
        rest = [r if (pd.notna(r) and r >= 1.0) else np.nan for r in rest]
        if all(pd.isna(r) for r in rest): continue
        rows.append({"Horse Number": int(uma), "単勝": rest[0], "複勝下限": rest[1], "複勝上限": rest[2]})
    return pd.DataFrame(rows, columns=["Horse Number", "単勝", "複勝下限", "複勝上限"]).drop_duplicates("Horse Number", keep="last")


def save_odds(sh, rid: str, race: pd.DataFrame, odds_df: pd.DataFrame, source="手入力/AI問い合わせ"):
    """odds_master へ Race ID 単位でUpsert（既存オッズは馬単位でマージ）。戻り値: 出走表に無い馬番のリスト。"""
    cur = load_odds(sh, rid, fresh=True)
    rec = {}
    if not cur.empty:
        for _, r in cur.iterrows(): rec[int(r["Horse Number"])] = (r.get("Win Odds"), r.get("Place Odds Min"), r.get("Place Odds Max"))
    names = {int(n): h for n, h in zip(race["Horse Number"], race["Horse Name"]) if pd.notna(n)}
    ignored = []
    for _, r in odds_df.iterrows():
        n = int(r["Horse Number"])
        if n not in names: ignored.append(n); continue
        old = rec.get(n, (np.nan, np.nan, np.nan))
        rec[n] = tuple(new if pd.notna(new) else o for new, o in zip((r["単勝"], r["複勝下限"], r["複勝上限"]), old))
    if not rec: raise ValueError("保存できるオッズがありません。")
    r0 = race.iloc[0]
    pop = pd.Series({n: v[0] for n, v in rec.items()}).rank(method="min")
    rows = [{"Race_Key_Src": rid, "Race Day": r0["Race Day"], "Venue": r0["Venue"], "Race Name": r0["Race Name"], "Horse Number": n,
             "Horse Name": names.get(n, ""), "Win Odds": v[0], "Place Odds Min": v[1], "Place Odds Max": v[2],
             "Popularity": pop.get(n, np.nan), "Odds Time": source, "Fetched At": now_str()} for n, v in sorted(rec.items())]
    upsert_by_race(sh, ODDS_SHEET, pd.DataFrame(rows), ODDS_COLUMNS)
    return ignored


def find_existing_ids(cur: pd.DataFrame, g: pd.DataFrame) -> set:
    """結果レースgに対応する既存Race ID: 同一ID、または『同日・同会場で馬名が6割以上一致』（手入力出走表とCSV結果のID不一致対策）。"""
    if cur.empty: return set()
    cids = race_ids_vec(cur)
    ids = set(cids[cids == g["_rid"].iloc[0]])
    names = set(g["Horse Name"].map(norm_name))
    same = cur[(cur["Race Day"] == g["Race Day"].iloc[0]) & (cur["Venue"] == g["Venue"].iloc[0])]
    for cid, x in same.groupby(cids[same.index]):
        if names and len(names & set(x["Horse Name"].map(norm_name))) / len(names) >= 0.6: ids.add(cid)
    return ids


def delete_management(sh, ids):
    if not ids: return
    try:
        ws = sh.worksheet(RACE_MANAGEMENT_SHEET)
    except gspread.exceptions.WorksheetNotFound:
        return
    col = ws.col_values(1)
    _delete_rows(ws, [i for i, v in enumerate(col, start=1) if i > 1 and v.strip() in ids])
    clear_sheet_cache()


def apply_results(sh, res: pd.DataFrame) -> dict:
    """確定結果(23列/任意CSV)を race_results と predictions に反映。AI予想列(AI_Probability等)は保持し、
    Placing / Last 3F / Passing / Tactics などの結果列だけ更新。該当レースが無ければ新規追加。race_management→結果登録済み。"""
    res = standardize_to_23cols(res)
    res = res[(res["Horse Name"] != "") & res["Placing"].notna()].copy()
    if res.empty: raise ValueError("着順の入った行がありません。")
    res["_rid"] = race_ids_vec(res); res["Race_ID"] = res["_rid"]
    if res["Passing"].str.strip().ne("").any():
        res["Tactics"] = derive_tactics(res).fillna(res["Tactics"])     # Passing(1〜2角)から脚質を判定
    stale, items = set(), []
    for name, cols in ((RESULTS_SHEET, RESULT_COLUMNS), (PREDICTIONS_SHEET, COLUMNS_23)):
        raw = read_sheet_df(sh, name, fresh=True)
        extra = [c for c in cols if c not in COLUMNS_23]
        cur = standardize_to_23cols(raw, extra_cols=extra) if not raw.empty else pd.DataFrame(columns=cols)
        merged, all_ex = [], set()
        for rid, g in res.groupby("_rid", sort=False):
            ex_ids = find_existing_ids(cur, g); all_ex |= ex_ids
            ex = cur[race_ids_vec(cur).isin(ex_ids)] if len(cur) else cur
            new = g.drop(columns=["_rid", "Race_ID"])
            src = str(new["Race_Key_Src"].iloc[0]); target = src if src else (sorted(ex_ids)[0] if ex_ids else rid)
            if ex.empty:
                m = new.copy()
            else:
                m = ex.copy(); m.index = m["Horse Name"].map(norm_name); m = m[~m.index.duplicated(keep="last")]
                nw = new.copy(); nw.index = nw["Horse Name"].map(norm_name); nw = nw[~nw.index.duplicated(keep="last")]
                for c in RESULT_FIELDS + ["Track Condition", "Weather"]:
                    v = nw[c].reindex(m.index)
                    m[c] = v.where(v.notna() & (v.astype(str).str.strip() != ""), m[c])
                m = pd.concat([m, nw[~nw.index.isin(m.index)]]).reset_index(drop=True)
            if re.fullmatch(r"\d{12}", str(target)): m["Race_Key_Src"] = target
            if name == RESULTS_SHEET: m["Race ID"] = target; m["Updated At"] = now_str()
            stale |= (ex_ids - set(race_ids_vec(m)))
            merged.append(m)
        allm = pd.concat(merged, ignore_index=True)
        upsert_by_race(sh, name, allm, cols, also_delete=all_ex)
        if name == RESULTS_SHEET: items = race_items_from_df(allm, STATUS_RESULT)
    upsert_management(sh, items)
    delete_management(sh, stale)
    return {"races": len(items), "horses": len(res)}


# ============================================================
# 履歴(Hist_*)の自動取得・補完：predictions / race_results から
# ============================================================
def load_horses_map(sh, fresh=False) -> dict:
    d = read_sheet_df(sh, HORSES_SHEET, fresh=fresh)
    m = {}
    if not d.empty and "Horse Name" in d.columns:
        for _, r in d.fillna("").iterrows():
            k = norm_name(r.get("Horse Name", ""))
            if k: m[k] = (str(r.get("Sire", "")).strip() or "Unknown", str(r.get("Dam's Sire", "")).strip() or "Unknown")
    return m


def load_history(sh, ped: dict, fresh=False) -> pd.DataFrame:
    """着順が入っている過去走だけを集め、特徴量計算用の内部列まで用意する。"""
    frames = []
    for t in (PREDICTIONS_SHEET, RESULTS_SHEET):                     # 後者(race_results)を優先
        d = read_sheet_df(sh, t, fresh=fresh)
        if d.empty or "Horse Name" not in d.columns: continue
        frames.append(standardize_to_23cols(d))
    if not frames: return pd.DataFrame()
    h = pd.concat(frames, ignore_index=True)
    h = h[h["Placing"].notna() & h["Horse Name"].ne("")].copy()
    if h.empty: return pd.DataFrame()
    h["Race_ID"] = race_ids_vec(h)
    h["Horse_Key"] = h["Horse Name"].map(norm_name)
    h = h.drop_duplicates(["Race_ID", "Horse_Key"], keep="last").reset_index(drop=True)
    h["Placing_num"] = h["Placing"]
    derived = derive_tactics(h)
    h["Tactics"] = derived.fillna(h["Tactics"])                       # Passing優先、無い行は元のTactics
    h["Sire"] = h["Horse_Key"].map(lambda k: ped.get(k, ("Unknown", "Unknown"))[0])
    h["Dam's Sire"] = h["Horse_Key"].map(lambda k: ped.get(k, ("Unknown", "Unknown"))[1])
    h = prep_columns(h)
    h["_tt"] = h["_tt"].fillna(-1)
    return h.sort_values("_day").reset_index(drop=True)


def get_builder(hist: pd.DataFrame, race_day):
    """レース日より前の履歴だけでFeatureBuilderを構築（同日・未来は含めない）。セッション内キャッシュ。"""
    key = (str(pd.Timestamp(race_day).date()), len(hist), str(hist["_day"].max()) if len(hist) else "")
    cache = st.session_state.setdefault("_fb_cache", {})
    if key in cache: return cache[key]
    fb = FeatureBuilder()
    if len(hist):
        for r in hist.loc[hist["_day"] < race_day, INTERNAL_COLS].to_dict("records"): fb.add(r)
    cache.clear(); cache[key] = fb
    return fb


def build_feature_rows(race: pd.DataFrame, hist: pd.DataFrame, ped: dict) -> pd.DataFrame:
    """1レース分の23列出走表 → モデル入力用の素の特徴量（馬名キーで Hist_* 等を自動補完）。"""
    r = race.copy().reset_index(drop=True)
    r["Horse_Key"] = r["Horse Name"].map(norm_name)
    r["Sire"] = r["Horse_Key"].map(lambda k: ped.get(k, ("Unknown", "Unknown"))[0])
    r["Dam's Sire"] = r["Horse_Key"].map(lambda k: ped.get(k, ("Unknown", "Unknown"))[1])
    r["Race_ID"] = race_ids_vec(r)
    r["Placing_num"] = np.nan; r["Last 3F"] = np.nan; r["Tactics"] = np.nan
    p = prep_columns(r); p["_tt"] = p["_tt"].fillna(-1)
    day = p["_day"].iloc[0]
    if pd.isna(day): raise ValueError("Race Day が解釈できないため履歴を参照できません。")
    fb = get_builder(hist, day) if len(hist) else FeatureBuilder()
    feats = pd.DataFrame([fb.features(x) for x in p[INTERNAL_COLS].to_dict("records")])
    out = pd.concat([r.drop(columns=["Last 3F", "Tactics", "Placing_num"]), feats], axis=1)
    out = add_member_relative_features(add_pace_features(out))
    return add_race_group_features(out)


# ============================================================
# モデル読込・予測
# ============================================================
@st.cache_resource(show_spinner="AIの頭脳を読み込み中…")
def load_model():
    if not MODEL_PATH.exists(): return None
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    if not (isinstance(obj, dict) and isinstance(obj.get("models"), dict) and all(k in obj["models"] for k in ("win", "top3"))):
        raise ValueError("model.pkl は train_kaggle.py が出力した形式（models に win / top3）ではありません。再学習してください。")
    return obj


GOING_ORDINAL = {"良": 0, "稍重": 1, "稍": 1, "重": 2, "不良": 3, "不": 3}  # unknown/missing is kept separate as -1
APP_BASE_FEATURES = ["Distance", "Bracket", "Horse Number", "Horse Age", "Burden Weight", "Track_Type", "Race Name", "Venue", "Track Condition", "Weather",
                     "Horse Name", "Sex", "Jockey", "Trainer", "Sire", "Dam's Sire",
                     "exp_count_nige", "exp_count_senko", "exp_count_sashi", "exp_count_oikomi", "exp_front_ratio", "field_size"]


def app_generated_features() -> set:
    """このアプリが生成できる特徴量名の全体。"""
    return set(APP_BASE_FEATURES) | set(BUILDER_FEATURES) | set(MEMBER_FEATURES) | set(SPECIAL_RACE_FEATURES) | set(PREP_FEATURES)


def missing_model_features(bundle) -> list:
    """モデルが使う特徴量のうち、アプリで生成できないもの（学習コードとアプリの特徴量定義のずれ）。"""
    if not bundle: return []
    ok = app_generated_features()
    return [f for f in bundle.get("features", []) if f not in ok]


def rows_to_matrix(rows: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    features = list(bundle["features"]); cat = set(bundle.get("categorical_features", []))
    enc = bundle.get("encoders", {}); nan_ok = set(bundle.get("nan_features", ["Bracket", "Horse Number"]))
    rows = rows.copy()
    for c in features:
        if c not in rows.columns: rows[c] = "Unknown" if c in cat else 0.0
    X = rows[features].copy()
    for c in features:
        if c in cat:
            X[c] = X[c].fillna("Unknown").astype(str).replace({"": "Unknown"}).map(enc.get(c, {})).fillna(-1).astype("int32")
        else:
            if c == "Track Condition":
                if not pd.api.types.is_numeric_dtype(X[c]):
                    v = X[c].astype(str).str.strip().map(GOING_ORDINAL)
                else:
                    v = pd.to_numeric(X[c], errors="coerce")
                X[c] = v.fillna(-1.0).astype(float)
            else:
                v = pd.to_numeric(X[c], errors="coerce")
                if c in ("Bracket", "Horse Number"):
                    v = v.where(v > 0)
                    X[c] = v.astype(float) if c in nan_ok else v.fillna(4.0 if c == "Bracket" else (len(X) + 1) / 2).astype(float)
                else:
                    X[c] = v.fillna(0).astype(float)
    return X


def predict_race(bundle, feats: pd.DataFrame):
    X = rows_to_matrix(feats, bundle)
    wm, tm = bundle["models"]["win"], bundle["models"]["top3"]
    win = np.asarray(wm.predict(X, num_iteration=getattr(wm, "best_iteration", None)), dtype=float)
    top3 = np.asarray(tm.predict(X, num_iteration=getattr(tm, "best_iteration", None)), dtype=float)
    # レース内で確率を正規化（勝率の合計=1、複勝率の合計=min(3,頭数)）
    n = len(win)
    win_n = win / max(win.sum(), 1e-9)
    k = min(3, n); top3_n = np.clip(top3 * (k / max(top3.sum(), 1e-9)), 0, 0.98)
    return win_n, top3_n


def pick_model(bundle, race: pd.DataFrame, scope="target"):
    """レースに応じて 長距離専用モデル / 汎用モデル を選ぶ。scope: target=菊花賞・天皇賞(春)・有馬記念のみ / all_long=2500m以上の平地すべて / off=使わない"""
    sp = (bundle.get("specialists") or {}).get("long") if bundle else None
    if not sp or scope == "off": return bundle, "汎用モデル"
    r0 = race.iloc[0]; group = race_group_code(r0["Race Name"]); dist, tt = r0["Distance"], r0["Track_Type"]
    long_flat = bool(pd.notna(dist) and dist >= sp.get("min_distance", 2500) and tt != 2)
    if (scope == "target" and group > 0) or (scope == "all_long" and long_flat):
        nm = SPECIAL_RACE_LABELS.get(group)
        return sp, "長距離専用モデル" + (f"（{nm}）" if nm else "")
    return bundle, "汎用モデル"


def model_is_current(bundle) -> bool:
    return bool(bundle) and bundle.get("feature_version") == FEATURE_VERSION


# ============================================================
# 詳細考察：展開予想 / 脚質分布 / 枠順バイアス
# ============================================================
def diagnose_pace(feats: pd.DataFrame) -> dict:
    code = pd.to_numeric(feats["Hist_Last_Tactics"], errors="coerce")
    code = code[code.between(1, 4)]
    total, known = len(feats), len(code)
    cnt = {i: int((code == i).sum()) for i in (1, 2, 3, 4)}
    base = {"total": total, "known": known, "n_nige": cnt[1], "n_senko": cnt[2], "n_sashi": cnt[3], "n_oikomi": cnt[4], "n_unknown": total - known}
    if known == 0: return {"ok": False, **base}
    front_ratio = (cnt[1] + cnt[2]) / known
    course = float(feats["Course_Front_Rate"].mean())             # このコースの過去の前残り傾向
    z = 10.0 * (front_ratio - 0.42) + 0.9 * (cnt[1] - 1.5) + 4.0 * (course - 0.35)
    high = float(np.clip(1 / (1 + np.exp(-z)), 0.01, 0.99)) * 100
    pace = "ハイペース" if high >= 60 else ("スローペース" if high <= 35 else "ミドルペース")
    return {"ok": True, "pace": pace, "high_prob": round(high, 1), "front_ratio": round(front_ratio * 100, 1),
            "course_front": round(course * 100, 1), **base}


def bracket_bias(hist: pd.DataFrame, race: pd.DataFrame, view: pd.DataFrame):
    """同競馬場×距離×芝ダの過去成績から、内/中/外枠の好走率とAI予測の枠別平均勝率を返す。"""
    r0 = race.iloc[0]; zones = ["内(1-2枠)", "中(3-5枠)", "外(6-8枠)"]
    res = {"table": None, "ai": None, "n": 0}
    if len(hist):
        m = (hist["_venue"] == norm_name(r0["Venue"])) & (hist["_dist"] == float(r0["Distance"] if pd.notna(r0["Distance"]) else -1))
        tt = r0["Track_Type"]
        if pd.notna(tt): m &= hist["_tt"] == float(tt)
        h = hist[m & hist["Bracket"].between(1, 8)].copy()
        if len(h):
            h["zone"] = pd.cut(h["Bracket"], [0, 2, 5, 8], labels=zones)
            t = h.groupby("zone", observed=False).agg(出走数=("_pl", "size"), 勝率=("_pl", lambda s: (s == 1).mean() * 100),
                                                       複勝率=("_pl", lambda s: (s <= 3).mean() * 100)).round(1)
            res["table"] = t; res["n"] = int(len(h))
    v = view[view["Bracket"].between(1, 8)].copy()
    if len(v):
        v["zone"] = pd.cut(v["Bracket"], [0, 2, 5, 8], labels=zones)
        res["ai"] = v.groupby("zone", observed=False).agg(頭数=("Horse Name", "size"), 平均勝率=("勝率(%)", "mean")).round(1)
    return res


def race_commentary(pace: dict, bias: dict, view: pd.DataFrame, feats: pd.DataFrame) -> list:
    out = []
    if pace.get("ok"):
        front = pace["n_nige"] + pace["n_senko"]
        nige_names = "・".join(feats.loc[feats["Hist_Last_Tactics"] == 1, "Horse Name"].tolist()) or "なし"
        if pace["pace"] == "ハイペース":
            txt = f"逃げ・先行が{front}頭（{pace['front_ratio']}%）と多く前が競り合う想定。**差し・追込が有利**になりやすい展開です。"
        elif pace["pace"] == "スローペース":
            txt = f"逃げ・先行は{front}頭（{pace['front_ratio']}%）と少なく楽に運べる想定。**逃げ・先行が有利**になりやすい展開です。"
        else:
            txt = f"逃げ・先行は{front}頭（{pace['front_ratio']}%）で平均的。**展開の偏りは小さく**、能力・位置取り次第です。"
        out.append(f"**① 展開予想：{pace['pace']}**（ハイペース確率 {pace['high_prob']}%）\n\n{txt}\n\n前走逃げた馬: {nige_names}　／　このコースの過去の前残り率: {pace['course_front']}%")
        top = view.head(3)
        fits = []
        for _, r in top.iterrows():
            t = r.get("前走脚質", "")
            if t in ("差し", "追込") and pace["pace"] == "ハイペース": fits.append(f"{r['Horse Name']}（{t}）は展開向き")
            elif t in ("逃げ", "先行") and pace["pace"] == "スローペース": fits.append(f"{r['Horse Name']}（{t}）は展開向き")
            elif t in ("逃げ", "先行") and pace["pace"] == "ハイペース": fits.append(f"{r['Horse Name']}（{t}）は展開に注意")
            elif t in ("差し", "追込") and pace["pace"] == "スローペース": fits.append(f"{r['Horse Name']}（{t}）は展開に注意")
        if fits: out.append("**AI上位馬と展開の相性**：" + "、".join(fits))
    else:
        out.append("**① 展開予想**：前走脚質が1頭も判明していないため判定できません（履歴データを蓄積してください）。")
    out.append(f"**② 脚質分布**：逃げ{pace.get('n_nige', 0)}／先行{pace.get('n_senko', 0)}／差し{pace.get('n_sashi', 0)}／追込{pace.get('n_oikomi', 0)}／不明{pace.get('n_unknown', 0)}頭")
    t = bias.get("table")
    if t is not None and bias["n"] >= 30:
        best = t["複勝率"].idxmax(); worst = t["複勝率"].idxmin(); gap = t["複勝率"].max() - t["複勝率"].min()
        if gap >= 4:
            out.append(f"**③ 枠順バイアス**（同コース過去{bias['n']}走）：**{best}の複勝率が{t.loc[best, '複勝率']}%と最も高く**、{worst}は{t.loc[worst, '複勝率']}%。{best}が有利、{worst}は割引の傾向です。")
        else:
            out.append(f"**③ 枠順バイアス**（同コース過去{bias['n']}走）：枠別の複勝率差は{gap:.1f}ptと小さく、**目立った有利不利はありません**。")
    else:
        out.append(f"**③ 枠順バイアス**：同コースの履歴が{bias['n']}走と少なく（30走未満）、統計的な判定は控えます。")
    return out


# ============================================================
# オッズ連動（odds_master を動的結合）→ 期待値 → 買い目自動算出
# ============================================================
def load_odds(sh, rid: str, fresh=False) -> pd.DataFrame:
    d = read_sheet_df(sh, ODDS_SHEET, fresh=fresh)
    if d.empty or "Race_Key_Src" not in d.columns: return pd.DataFrame()
    d = d[d["Race_Key_Src"].astype(str).str.replace(r"\.0$", "", regex=True) == str(rid)].copy()
    for c in ("Horse Number", "Win Odds", "Place Odds Min", "Place Odds Max", "Popularity"):
        if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.drop_duplicates("Horse Number", keep="last")


def attach_odds(view: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    v = view.drop(columns=[c for c in ("単勝オッズ", "複勝オッズ(下限)", "人気", "期待値(単勝)", "期待値(複勝)") if c in view.columns]).copy()
    if odds is None or odds.empty:
        for c in ("単勝オッズ", "複勝オッズ(下限)", "人気", "期待値(単勝)", "期待値(複勝)"): v[c] = np.nan
        return v
    o = odds.rename(columns={"Win Odds": "単勝オッズ", "Place Odds Min": "複勝オッズ(下限)", "Popularity": "人気"})
    v = v.merge(o[["Horse Number", "単勝オッズ", "複勝オッズ(下限)", "人気"]], on="Horse Number", how="left")
    v["期待値(単勝)"] = (v["勝率(%)"] / 100 * v["単勝オッズ"]).round(2)          # 勝率 × オッズ
    v["期待値(複勝)"] = (v["複勝率(%)"] / 100 * v["複勝オッズ(下限)"]).round(2)    # 複勝率 × 複勝オッズ下限（保守的）
    return v


def allocate_units(weights, total_units):
    """重みに比例して100円単位(=unit)で按分。各買い目は最低1unit。足りない場合は重みの小さい買い目から落とす。"""
    idx = list(range(len(weights)))
    while len(idx) > total_units:
        idx.remove(min(idx, key=lambda i: weights[i]))
    if not idx: return {}
    units = {i: 1 for i in idx}; rest = total_units - len(idx)
    w = np.array([max(weights[i], 1e-9) for i in idx]); share = w / w.sum() * rest
    for i, s in zip(idx, share): units[i] += int(np.floor(s))
    left = total_units - sum(units.values())
    for i, _ in sorted(zip(idx, share - np.floor(share)), key=lambda x: -x[1])[:left]: units[i] += 1
    return units


def make_bets(v: pd.DataFrame, budget: int, strategy: str):
    """戻り値: (買い目リスト, メッセージ)。対象は odds_master にある 単勝・複勝（他券種のオッズは未保有のため対象外）。"""
    total_units = int(budget // 100)
    if total_units < 1: return [], "予算は100円以上で入力してください。"
    d = v.dropna(subset=["単勝オッズ"]).copy()
    if d.empty: return [], "odds_master に該当レースのオッズがありません（タブ1の「オッズ入力」でAI回答の貼り付けまたは手入力をしてください）。"
    cands, note = [], ""
    if strategy.startswith("的中"):
        for _, r in d.sort_values("複勝率(%)", ascending=False).head(3).iterrows():
            cands.append(("複勝", r, r["複勝率(%)"] / 100, r["複勝オッズ(下限)"], r["複勝率(%)"] / 100))
        for _, r in d.sort_values("勝率(%)", ascending=False).head(1).iterrows():
            cands.append(("単勝", r, r["勝率(%)"] / 100, r["単勝オッズ"], r["勝率(%)"] / 100 * 0.6))
    else:
        for _, r in d.iterrows():
            if pd.notna(r["期待値(単勝)"]) and r["期待値(単勝)"] >= 1.0:
                cands.append(("単勝", r, r["勝率(%)"] / 100, r["単勝オッズ"], (r["期待値(単勝)"] - 1) * r["勝率(%)"] / 100))
            if pd.notna(r["期待値(複勝)"]) and r["期待値(複勝)"] >= 1.0:
                cands.append(("複勝", r, r["複勝率(%)"] / 100, r["複勝オッズ(下限)"], (r["期待値(複勝)"] - 1) * r["複勝率(%)"] / 100))
        cands = sorted(cands, key=lambda c: -c[4])[:4]
        if not cands:
            r = d.sort_values("期待値(単勝)", ascending=False).iloc[0]
            cands.append(("単勝", r, r["勝率(%)"] / 100, r["単勝オッズ"], 1.0))
            note = "期待値1.0以上の馬券がないため、最も期待値の高い単勝を1点のみ提示します（見送りも選択肢です）。"
    units = allocate_units([c[4] for c in cands], total_units)
    bets = []
    for i, (kind, r, p, odds, _) in enumerate(cands):
        if i not in units: continue
        amt = units[i] * 100
        bets.append({"種別": kind, "馬番": int(r["Horse Number"]), "馬名": r["Horse Name"], "金額": amt, "オッズ": float(odds),
                     "的中確率(%)": round(p * 100, 1), "期待払戻": round(amt * p * float(odds))})
    return bets, note


# ============================================================
# 予想結果の保存（Upsertのみ）＆ 出走表バッファのリセット
# ============================================================
def _carry_results(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """既に確定結果(Placing)がある場合は、再予想しても結果列を失わないよう馬番で引き継ぐ。"""
    if existing.empty or "Placing" not in existing.columns: return new
    e = standardize_to_23cols(existing)
    e = e[e["Placing"].notna()].drop_duplicates("Horse Number").set_index("Horse Number")
    if e.empty: return new
    n = new.copy()
    for c in RESULT_FIELDS:
        n[c] = n["Horse Number"].map(e[c]).where(n["Horse Number"].map(e["Placing"]).notna(), n[c])
    return n


def save_prediction(sh, rid, race23: pd.DataFrame, view: pd.DataFrame, pace: dict, top3_map: dict, clear_mode: str):
    """predictions と race_results をUpsert → race_management を『予想済み』 → 出走表バッファをリセット。"""
    out = race23.copy()
    wp = view.set_index("Horse Name")["勝率(%)"]; rk = view.set_index("Horse Name")["予想順位"]
    out["AI_Probability"] = out["Horse Name"].map(wp); out["AI_Rank"] = out["Horse Name"].map(rk)
    out["Race_Key_Src"] = out["Race_Key_Src"].where(out["Race_Key_Src"] != "", rid)
    for name, cols in ((PREDICTIONS_SHEET, COLUMNS_23), (RESULTS_SHEET, RESULT_COLUMNS)):
        cur = read_sheet_df(sh, name, fresh=True)
        if not cur.empty:
            cur = cur[race_ids_vec(cur) == rid]
        o = _carry_results(cur, out)
        if name == RESULTS_SHEET:
            o["Race ID"] = rid
            o["Top3 Probability"] = o["Horse Name"].map(top3_map)
            o["Pace"] = pace.get("pace", ""); o["High Pace Prob"] = pace.get("high_prob", ""); o["Front Ratio"] = pace.get("front_ratio", "")
            o["Predicted At"] = now_str(); o["Updated At"] = now_str()
        upsert_by_race(sh, name, o, cols)
    upsert_management(sh, race_items_from_df(out, STATUS_PREDICTED))
    if clear_mode == "all":
        ws = sh.worksheet(ENTRY_SHEET)
        ws.batch_clear([f"A2:{_col_letter(max(ws.col_count, 1))}{max(ws.row_count, 2)}"])
        clear_sheet_cache()
    elif clear_mode == "race":
        delete_races(sh, ENTRY_SHEET, [rid])


# ============================================================
# 検証：23列の厳格チェック ＆ 予想 vs 実績
# ============================================================
def check_23_completeness(df: pd.DataFrame, expected_rids):
    """戻り値: (OKか, 問題リスト)。レース欠落・23列の空欄を厳格に検出（'**'は空欄扱い）。"""
    problems = []
    if df.empty:
        return False, ["race_results に対象データがありません。"]
    raw = df.copy(); raw["_rid"] = race_ids_vec(raw)
    for c in COLUMNS_23:
        if c not in raw.columns: problems.append(f"列そのものが存在しません: {c}")
    present = set(raw["_rid"])
    for rid in expected_rids:
        if rid not in present: problems.append(f"{rid}: 確定結果の行がありません")
    for c in COLUMNS_23:
        if c in raw.columns and c not in VERIFY_OPTIONAL_COLS:
            blank = raw[c].astype(str).str.strip().replace({"**": "", "nan": "", "None": ""}).eq("")
            if blank.any():
                bad = raw.loc[blank, "_rid"].value_counts()
                problems.append(f"{c}: 空欄 {int(blank.sum())}件（{', '.join(f'{k}×{v}' for k, v in bad.head(5).items())}{' …' if len(bad) > 5 else ''}）")
    return (len(problems) == 0), problems


def build_verification(res: pd.DataFrame, odds_all: pd.DataFrame) -> pd.DataFrame:
    d = standardize_to_23cols(res)
    d["_rid"] = race_ids_vec(d)
    od = odds_all.copy() if odds_all is not None else pd.DataFrame()
    if not od.empty:
        od["_rid"] = od["Race_Key_Src"].astype(str).str.replace(r"\.0$", "", regex=True)
        for c in ("Horse Number", "Win Odds", "Place Odds Min"): od[c] = pd.to_numeric(od[c], errors="coerce")
        od = od.drop_duplicates(["_rid", "Horse Number"], keep="last").set_index(["_rid", "Horse Number"])
    rows = []
    for rid, g in d.groupby("_rid", sort=True):
        g = g.sort_values("AI_Rank"); top = g.iloc[0]; r0 = g.iloc[0]
        actual3 = set(g.loc[g["Placing"] <= 3, "Horse Number"]); ai3 = set(g.head(3)["Horse Number"])
        win_hit = bool(top["Placing"] == 1); place_hit = bool(top["Placing"] <= 3)
        wo = po = np.nan
        if not od.empty and (rid, top["Horse Number"]) in od.index:
            wo = od.loc[(rid, top["Horse Number"]), "Win Odds"]; po = od.loc[(rid, top["Horse Number"]), "Place Odds Min"]
        rows.append({"Race ID": rid, "レース": f"{r0['Venue']} {r0['Race Name']}", "AI本命": top["Horse Name"], "本命着順": int(top["Placing"]),
                     "単勝的中": win_hit, "複勝的中": place_hit, "AI上位3頭中の3着内": len(ai3 & actual3),
                     "単勝オッズ": wo, "単勝払戻(100円)": (wo * 100 if win_hit and pd.notna(wo) else (0.0 if pd.notna(wo) else np.nan)),
                     "複勝オッズ(下限)": po, "複勝払戻(100円)": (po * 100 if place_hit and pd.notna(po) else (0.0 if pd.notna(po) else np.nan))})
    return pd.DataFrame(rows)


# ============================================================
# 出走表の絞り込み・表示用ヘルパ
# ============================================================
def filter_entries(df: pd.DataFrame, q: str) -> pd.DataFrame:
    """『10月25日』『2026-10-25』『菊花賞』『中山』などで絞り込み（日付とレース名は併用可）。"""
    q = unicodedata.normalize("NFKC", (q or "")).strip()
    if not q or df.empty: return df
    m = df["Race Day"].astype(str).str.len() > 0
    md = re.search(r"(\d{1,2})月(\d{1,2})日", q); iso = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", q)
    days = pd.to_datetime(df["Race Day"], errors="coerce")
    rest = q
    if iso:
        m &= days.dt.strftime("%Y-%m-%d") == f"{int(iso.group(1)):04d}-{int(iso.group(2)):02d}-{int(iso.group(3)):02d}"; rest = rest.replace(iso.group(0), "")
    elif md:
        m &= (days.dt.month == int(md.group(1))) & (days.dt.day == int(md.group(2))); rest = rest.replace(md.group(0), "")
    for tok in rest.split():
        m &= df["Race Name"].astype(str).str.contains(tok, regex=False) | df["Venue"].astype(str).str.contains(tok, regex=False)
    return df[m]


def race_label(rid: str, g: pd.DataFrame) -> str:
    r = g.iloc[0]; rno = ""
    if re.fullmatch(r"\d{12}", str(rid)): rno = f"{int(str(rid)[-2:])}R"
    tt = {0: "芝", 1: "ダ", 2: "障"}.get(r["Track_Type"], "")
    dist = f"{int(r['Distance'])}" if pd.notna(r["Distance"]) else "?"
    return f"{r['Venue']}{rno} {r['Race Name']}（{tt}{dist}m・{len(g)}頭）"


# ============================================================
# 過去予想の呼び出し ／ 当日条件の変更（シナリオ）予想
#   ・変更できるもの: 馬場状態・天候・出走取消・枠番/馬番・騎手・斤量 → モデルへの入力そのものを変えて再推論
#   ・ペース: モデルは『ペース』を入力に持たないため、履歴から求めた『ペース×脚質』の有利不利倍率で事後補正（脚質の想定は馬ごとに変更可）
# ============================================================
GOING_OPTIONS = ["良", "稍重", "重", "不良"]
WEATHER_OPTIONS = ["晴", "曇", "小雨", "雨", "小雪", "雪"]
PACE_NAMES = ["ハイペース", "ミドルペース", "スローペース"]
PACE_CHOICES = ["自動（補正なし）"] + PACE_NAMES
TAC_AUTO = "自動（前走）"
TAC_CODE = {"逃げ": 1, "先行": 2, "差し": 3, "追込": 4}
PACE_HIGH_PROB = {"ハイペース": 85.0, "ミドルペース": 50.0, "スローペース": 15.0}
# 履歴が少ない場合の既定倍率（勝率・複勝率とも同じ。目安であり、実データから算出できる場合はそちらを優先）
PACE_DEFAULT_LIFT = {"ハイペース": {1: 0.85, 2: 0.93, 3: 1.08, 4: 1.15},
                     "ミドルペース": {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
                     "スローペース": {1: 1.20, 2: 1.10, 3: 0.92, 4: 0.82}}


def pace_tactic_lifts(hist: pd.DataFrame, min_races=300, shrink=200.0):
    """履歴から『ペース(場の逃げ先行割合で3分類) × 脚質』ごとの 勝率・複勝率 の倍率（全体比）を算出。件数が少なければ既定値。"""
    default = {"win": PACE_DEFAULT_LIFT, "top3": PACE_DEFAULT_LIFT, "source": "既定値（履歴が少ないため）"}
    try:
        if hist is None or len(hist) == 0: return default
        h = hist[pd.to_numeric(hist["_tac"], errors="coerce").between(1, 4) & hist["_pl"].notna()].copy()
        h["_front"] = (h["_tac"] <= 2).groupby(h["Race_ID"]).transform("mean")
        h = h[h.groupby("Race_ID")["_tac"].transform("size") >= 8]
        n_races = h["Race_ID"].nunique()
        if n_races < min_races: return default
        races = h.drop_duplicates("Race_ID")
        hi, lo = races["_front"].quantile(0.7), races["_front"].quantile(0.3)
        h["_pace"] = np.where(h["_front"] >= hi, "ハイペース", np.where(h["_front"] <= lo, "スローペース", "ミドルペース"))
        out = {"source": f"履歴 {n_races:,}レースから算出（場の逃げ・先行割合で3分類）"}
        for kind, cond in (("win", h["_pl"] == 1), ("top3", h["_pl"] <= 3)):
            h["_y"] = cond.astype(float); overall = h.groupby("_tac")["_y"].mean(); out[kind] = {}
            for p in PACE_NAMES:
                agg = h[h["_pace"] == p].groupby("_tac")["_y"].agg(["sum", "count"])
                out[kind][p] = {}
                for t in (1, 2, 3, 4):
                    if t in agg.index and overall.get(t, 0) > 0:
                        rate = (agg.loc[t, "sum"] + shrink * overall[t]) / (agg.loc[t, "count"] + shrink)
                        out[kind][p][t] = float(np.clip(rate / overall[t], 0.5, 1.8))
                    else:
                        out[kind][p][t] = 1.0
        return out
    except Exception:
        return default


def apply_pace_adjustment(win, top3, tactics, pace, lifts, strength=1.0):
    """想定脚質(1〜4, 0=不明)に応じて 勝率・複勝率 を補正し、再び 合計=1 / 合計=min(3,頭数) に正規化。"""
    lw = np.array([lifts["win"][pace].get(int(t), 1.0) for t in tactics], dtype=float) ** strength
    l3 = np.array([lifts["top3"][pace].get(int(t), 1.0) for t in tactics], dtype=float) ** strength
    w = np.asarray(win, dtype=float) * lw; w = w / max(w.sum(), 1e-9)
    n = len(w); k = min(3, n)
    t3 = np.clip(np.asarray(top3, dtype=float) * l3, 0, 0.98); t3 = np.clip(t3 * (k / max(t3.sum(), 1e-9)), 0, 0.98)
    return w, t3


def build_edit_table(race: pd.DataFrame) -> pd.DataFrame:
    n = len(race)
    return pd.DataFrame({"出走": [True] * n, "Horse Number": race["Horse Number"].astype("Int64").values, "Horse Name": race["Horse Name"].values,
                         "Bracket": race["Bracket"].astype("Int64").values, "Jockey": race["Jockey"].values,
                         "Burden Weight": race["Burden Weight"].values, "想定脚質": [TAC_AUTO] * n})


def apply_overrides(race: pd.DataFrame, going=None, weather=None, edits: pd.DataFrame = None):
    """当日変化しうる要素をレース行へ反映。戻り値: (変更後のレース, {馬名: 想定脚質コード})。"""
    r = race.copy().reset_index(drop=True)
    if going: r["Track Condition"] = going
    if weather: r["Weather"] = weather
    tac = {}
    if edits is not None and len(edits):
        e = edits.reset_index(drop=True)
        if len(e) != len(r): raise ValueError("出走馬の編集表の行数がレースと一致しません。")
        for col in ("Bracket", "Horse Number", "Burden Weight"): r[col] = pd.to_numeric(e[col], errors="coerce").values
        r["Jockey"] = e["Jockey"].fillna("").astype(str).values
        tac = {n: TAC_CODE[t] for n, t in zip(e["Horse Name"], e["想定脚質"]) if t in TAC_CODE}
        r = r[e["出走"].fillna(True).astype(bool).values].reset_index(drop=True)
    if len(r) < 2: raise ValueError("出走馬が2頭未満です。")
    return r, tac


def describe_scenario(orig: pd.DataFrame, new: pd.DataFrame, going, weather, pace_choice, tac_over) -> str:
    parts = []
    o = orig.iloc[0]
    if going and going != o["Track Condition"]: parts.append(f"馬場:{going}")
    if weather and weather != o["Weather"]: parts.append(f"天候:{weather}")
    if pace_choice in PACE_NAMES: parts.append(pace_choice)
    gone = set(orig["Horse Name"]) - set(new["Horse Name"])
    if gone: parts.append(f"取消{len(gone)}頭")
    m = orig.set_index("Horse Name"); nn = new.set_index("Horse Name"); common = [h for h in nn.index if h in m.index]
    if any(str(m.loc[h, "Jockey"]) != str(nn.loc[h, "Jockey"]) for h in common): parts.append("騎手変更")
    if any(not (pd.isna(m.loc[h, "Burden Weight"]) and pd.isna(nn.loc[h, "Burden Weight"])) and m.loc[h, "Burden Weight"] != nn.loc[h, "Burden Weight"] for h in common): parts.append("斤量変更")
    if any(not (pd.isna(m.loc[h, "Horse Number"]) and pd.isna(nn.loc[h, "Horse Number"])) and (m.loc[h, "Horse Number"] != nn.loc[h, "Horse Number"] or m.loc[h, "Bracket"] != nn.loc[h, "Bracket"]) for h in common): parts.append("枠順変更")
    if tac_over: parts.append(f"脚質想定{len(tac_over)}頭")
    return "／".join(parts) or "基準（入力どおり）"


def run_scenario(bundle, hist, ped, race, spec_scope="target", going=None, weather=None, edits=None, pace_choice=None, strength=1.0, lifts=None):
    """条件を反映して推論。画面に依存しない（テスト可能）。"""
    r, tac_over = apply_overrides(race, going, weather, edits)
    label = describe_scenario(race, r, going, weather, pace_choice, tac_over)
    feats = build_feature_rows(r, hist, ped)
    use_bundle, model_label = pick_model(bundle, r, spec_scope)
    win, top3 = predict_race(use_bundle, feats)
    raw_win = win.copy()
    tactics = pd.to_numeric(feats["Hist_Last_Tactics"], errors="coerce").where(lambda s: s.isin([1, 2, 3, 4]), 0).fillna(0).astype(int).values.copy()
    for i, n in enumerate(feats["Horse Name"]):
        if n in tac_over: tactics[i] = tac_over[n]                    # 想定脚質（ペース補正・展開分析にのみ使用。モデル入力は変えない）
    adjusted = pace_choice in PACE_NAMES
    if adjusted:
        lifts = lifts or {"win": PACE_DEFAULT_LIFT, "top3": PACE_DEFAULT_LIFT, "source": "既定値"}
        win, top3 = apply_pace_adjustment(win, top3, tactics, pace_choice, lifts, strength)
    feats_view = feats.copy(); feats_view["Hist_Last_Tactics"] = tactics.astype(float)
    view = pd.DataFrame({
        "Horse Number": feats["Horse Number"], "Horse Name": feats["Horse Name"], "Bracket": feats["Bracket"], "Jockey": feats["Jockey"],
        "勝率(%)": np.round(win * 100, 1), "複勝率(%)": np.round(top3 * 100, 1),
        "前走脚質": feats["Hist_Last_Tactics"].map(lambda x: TACTICS_LABEL.get(int(x), "-") if x in (1, 2, 3, 4) else "-"),
        "想定脚質": pd.Series(tactics).map(lambda x: TACTICS_LABEL.get(int(x), "-")).values,
        "前走上がり3F": feats["Hist_Last3F_Last"].where(feats["Hist_Last3F_Last"] > 0),
        "前走着差(相対着順)": feats["Rel_Last_Margin"].where(feats["Rel_Last_Margin"] >= 0).round(2),
        "間隔(日)": feats["Hist_Rest_Days"].where(feats["Hist_Rest_Days"] >= 0)})
    if adjusted: view["補正前勝率(%)"] = np.round(raw_win * 100, 1)
    view["予想順位"] = view["勝率(%)"].rank(ascending=False, method="min").astype(int)
    view = view.sort_values("予想順位").reset_index(drop=True)
    pace = diagnose_pace(feats_view)
    if adjusted:
        pace = dict(pace); pace.update({"ok": True, "pace": pace_choice, "high_prob": PACE_HIGH_PROB[pace_choice], "user_override": True})
        pace.setdefault("front_ratio", 0.0); pace.setdefault("course_front", 0.0)
    bias = bracket_bias(hist, r, view)
    return {"label": label, "race": r, "view": view, "feats": feats_view, "pace": pace, "bias": bias, "model_label": model_label,
            "missing_features": missing_model_features(use_bundle),
            "comments": race_commentary(pace, bias, view, feats_view), "pace_choice": pace_choice if adjusted else None,
            "lift_source": (lifts or {}).get("source", "") if adjusted else "", "strength": strength, "saved": False}


def compare_scenarios(scens: dict, base_label: str) -> pd.DataFrame:
    """馬ごとに 各シナリオの勝率・順位 と 基準シナリオとの差 を並べる。"""
    base = scens[base_label]["view"].set_index("Horse Name")
    names = list(dict.fromkeys(h for s in scens.values() for h in s["view"]["Horse Name"]))
    out = pd.DataFrame({"Horse Name": names})
    out.insert(0, "Horse Number", [next((int(s["view"].set_index("Horse Name").loc[h, "Horse Number"]) for s in scens.values() if h in set(s["view"]["Horse Name"]) and pd.notna(s["view"].set_index("Horse Name").loc[h, "Horse Number"])), None) for h in names])
    for lab, s in scens.items():
        v = s["view"].set_index("Horse Name")
        out[f"{lab}｜勝率(%)"] = out["Horse Name"].map(v["勝率(%)"]); out[f"{lab}｜順位"] = out["Horse Name"].map(v["予想順位"])
        if lab != base_label: out[f"{lab}｜Δ勝率(vs基準)"] = (out[f"{lab}｜勝率(%)"] - out["Horse Name"].map(base["勝率(%)"])).round(1)
    first = f"{base_label}｜順位"
    return out.sort_values(first, na_position="last").reset_index(drop=True)


def past_race_options(sh, q="") -> pd.DataFrame:
    """過去に予想した（予想済み／結果登録済み）レースの一覧（新しい順）。q: 『10月25日』『菊花賞』などで絞り込み。"""
    mg = load_race_management_df(sh, fresh=False)
    mg = mg[mg["Status"].isin([STATUS_PREDICTED, STATUS_RESULT])].copy()
    if mg.empty: return mg
    if q.strip():
        mg = filter_entries(mg, q)
    mg["_label"] = mg["Race Day"].astype(str) + " " + mg["Venue"].astype(str) + " " + mg["Race Name"].astype(str) + "（" + mg["Status"].astype(str) + "）"
    return mg.sort_values(["Race Day", "Race ID"], ascending=[False, True]).reset_index(drop=True)


def load_past_race(sh, rid: str) -> pd.DataFrame:
    """race_results（無ければ predictions）から、そのレースの行（AI予想・結果列を含む）を23列形式で取得。"""
    for name, extra in ((RESULTS_SHEET, [c for c in RESULT_COLUMNS if c not in COLUMNS_23]), (PREDICTIONS_SHEET, [])):
        raw = read_sheet_df(sh, name)
        if raw.empty: continue
        src = raw[race_ids_vec(raw) == rid]
        if not src.empty:
            return standardize_to_23cols(src, extra_cols=extra).sort_values("Horse Number").reset_index(drop=True)
    return pd.DataFrame()


def saved_prediction_view(r: pd.DataFrame) -> pd.DataFrame:
    """保存済みの予想結果（AI_Probability / AI_Rank / Top3 Probability）と、結果があれば着順を一覧にする。"""
    t3 = pd.to_numeric(r["Top3 Probability"], errors="coerce") if "Top3 Probability" in r.columns else pd.Series(np.nan, index=r.index)
    out = pd.DataFrame({"予想順位": r["AI_Rank"], "Horse Number": r["Horse Number"], "Horse Name": r["Horse Name"], "勝率(%)": r["AI_Probability"],
                        "複勝率(%)": t3, "結果(着順)": r["Placing"], "脚質(実績)": r["Tactics"].map(lambda x: TACTICS_LABEL.get(int(x), "-") if pd.notna(x) else "-")})
    return out.sort_values("予想順位", na_position="last").reset_index(drop=True)



# ============================================================
# 画面
# ============================================================
st.title("🏇 競馬予想AI オールインワン（23列仕様）")
S = st.session_state
S.setdefault("entries", pd.DataFrame()); S.setdefault("preds", {}); S.setdefault("bets", {})

try:
    sh = get_spreadsheet()
except Exception as exc:
    st.error(f"スプレッドシートに接続できません: {exc}"); st.stop()
try:
    bundle = load_model()
except Exception as exc:
    bundle = None; st.error(f"モデル読込エラー: {exc}")


def get_hist(force=False):
    if force or "hist" not in S:
        with st.spinner("predictions / race_results から履歴を読み込み中…"):
            S["_fb_cache"] = {}
            S["ped"] = load_horses_map(sh, fresh=force)
            S["hist"] = load_history(sh, S["ped"], fresh=force)
    return S["hist"], S["ped"]


with st.sidebar:
    st.header("🤖 モデル")
    if bundle is None:
        st.warning("model.pkl がありません。train_kaggle.py で学習して配置してください。")
    else:
        st.caption(f"期間 {bundle.get('date_min')}〜{bundle.get('date_max')} / {bundle.get('rows_total', 0):,}行 / 特徴量{len(bundle['features'])}個")
        _miss = missing_model_features(bundle)
        _miss_sp = missing_model_features((bundle.get("specialists") or {}).get("long"))
        if _miss or _miss_sp:
            st.error(f"このモデルが使う特徴量のうち、アプリで生成できないものがあります: {sorted(set(_miss + _miss_sp))}。値0で代用して推論するため精度が落ちます。学習コードの特徴量定義とアプリ(app.py)を合わせてください。")
        elif model_is_current(bundle): st.success(f"特徴量バージョン一致（{FEATURE_VERSION}）")
        else: st.info(f"モデルの特徴量バージョン {bundle.get('feature_version', '不明')}（アプリ {FEATURE_VERSION}）。必要な特徴量はすべて生成できます。")
        if "Track Condition" in bundle.get("numeric_features", []):
            st.caption("馬場状態は数値（良0・稍重1・重2・不良3）で使用（推定。学習コードの定義と一致しているか要確認）")
    if st.button("🔄 履歴・血統を再読込", **STRETCH):
        get_hist(force=True); st.success("履歴を再読込しました")
    _sp = (bundle.get("specialists") or {}).get("long") if bundle else None
    if _sp:
        _m = _sp.get("metrics", {})
        st.caption(f"長距離専用モデル: あり（{_sp.get('min_distance', 2500)}m以上・学習{_m.get('rows_train', 0):,}行 / 検証AUC 単勝{_m.get('auc_win_spec', float('nan')):.3f}）")
    if "hist" in S: st.caption(f"履歴: {len(S['hist']):,}走 / 血統登録 {len(S.get('ped', {})):,}頭")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["① 予想実行・買い目", "② 予想 vs 実績検証", "③ CSV同期・データ入力", "④ レース管理", "⑤ 血統マスタ"])

# ------------------------------------------------------------------ タブ1
def get_lifts(hist):
    key = len(hist)
    if S.get("_lift_key") != key:
        S["_lifts"] = pace_tactic_lifts(hist); S["_lift_key"] = key
    return S["_lifts"]


with tab1:
    S.setdefault("scen", {})
    src_mode = st.radio("レースの呼び出し元", ["出走表から新規予想", "過去に予想したレースを呼び出す"], horizontal=True, key="t1_src")
    q = st.text_input("日付・レース名で絞り込み", placeholder="例: 10月25日 / 菊花賞 / 2026-10-25 中山", key="q_entry")
    race = None; rid = None; day = ""; label_txt = ""; is_past = False
    if src_mode.startswith("出走表"):
        st.subheader("1. 出走表の読み込み")
        st.caption("出走表はタブ③で『手入力 / 貼り付け / CSVアップロード』して「出走表」タブへ保存してから、ここで読み込みます。")
        if st.button("📥 出走表シートを読み込む（全レース）", type="primary"):
            raw = read_sheet_df(sh, ENTRY_SHEET, fresh=True)
            if raw.empty:
                st.warning("出走表タブが空です。タブ③で出走表を手入力／貼り付け／CSV同期してください。")
            else:
                ent = standardize_to_23cols(raw)
                ent = ent[ent["Horse Name"] != ""].copy(); ent["_rid"] = race_ids_vec(ent)
                S["entries"] = ent
                try: upsert_management(sh, race_items_from_df(ent, STATUS_UNPREDICTED))
                except Exception as exc: st.warning(f"race_management への未予想登録に失敗: {exc}")
                st.success(f"{ent['_rid'].nunique()}レース / {len(ent)}頭を読み込みました（新規レースは『未予想』で登録）")
        ent = S["entries"]
        if ent.empty:
            st.info("出走表を読み込むとレース選択が表示されます。")
        else:
            fe = filter_entries(ent, q)
            if fe.empty:
                st.warning("該当するレースがありません。")
            else:
                day = st.selectbox("日付", sorted(fe["Race Day"].unique()), key="sel_day")
                fd = fe[fe["Race Day"] == day]
                groups = {r_: g for r_, g in fd.groupby("_rid", sort=True)}
                labels = {r_: race_label(r_, g) for r_, g in groups.items()}
                rid = st.selectbox("レース", list(groups), format_func=lambda x: labels[x], key="sel_race")
                race = groups[rid].sort_values("Horse Number").reset_index(drop=True); label_txt = labels[rid]
    else:
        is_past = True
        st.subheader("1. 過去に予想したレースを選ぶ")
        opts = past_race_options(sh, q)
        if opts.empty:
            st.info("予想済みのレースがありません（条件に合うレースが無い場合は、絞り込みを変えてください）。")
        else:
            lab_map = dict(zip(opts["Race ID"], opts["_label"]))
            rid = st.selectbox("レース", list(lab_map), format_func=lambda x: lab_map[x], key="sel_past")
            saved = load_past_race(sh, rid)
            if saved.empty:
                st.warning("このレースの行が race_results / predictions にありません。")
            else:
                race = saved.copy(); day = str(race["Race Day"].iloc[0]); label_txt = lab_map[rid]
                st.markdown("**保存済みの予想結果**")
                st.dataframe(saved_prediction_view(saved), hide_index=True, **STRETCH)
                r0 = saved.iloc[0]
                bits = [f"{k}: {r0[k]}" for k in ("Pace", "High Pace Prob", "Front Ratio", "Predicted At") if k in saved.columns and str(r0.get(k, "")).strip() not in ("", "nan")]
                if bits: st.caption("／".join(bits))
                if saved["Placing"].notna().any():
                    top = saved.sort_values("AI_Rank").iloc[0]
                    st.caption(f"結果登録済み：AI本命は {top['Horse Name']}（結果 {int(top['Placing']) if pd.notna(top['Placing']) else '-'}着）")

    if race is not None:
        st.subheader("2. 当日条件の変更（シナリオ）")
        st.caption("重馬場・天候・出走取消・枠順・騎手・斤量はモデルの入力を変えて再推論します。ペースはモデルの入力に無いため、履歴から求めた『ペース×脚質』の有利不利倍率で補正します（想定脚質は馬ごとに変更可）。")
        r0 = race.iloc[0]
        c1, c2, c3 = st.columns(3)
        going_sel = c1.selectbox("馬場状態", [f"入力どおり（{r0['Track Condition'] or '不明'}）"] + GOING_OPTIONS, key=f"sc_{rid}_going")
        weather_sel = c2.selectbox("天候", [f"入力どおり（{r0['Weather'] or '不明'}）"] + WEATHER_OPTIONS, key=f"sc_{rid}_weather")
        pace_sel = c3.selectbox("ペース", PACE_CHOICES, key=f"sc_{rid}_pace")
        strength = st.slider("ペース補正の強さ（1.0=履歴どおり／0=補正なし）", 0.0, 1.5, 1.0, 0.1, key=f"sc_{rid}_str", disabled=pace_sel not in PACE_NAMES)
        lift_mode = st.radio("ペース補正の倍率", ["履歴から算出", "定石の既定値"], horizontal=True, key=f"sc_{rid}_liftmode", disabled=pace_sel not in PACE_NAMES,
                             help="履歴から算出: 保存済みの過去成績で『場の逃げ・先行割合×脚質』ごとの勝率・複勝率の全体比を求めた値。件数が少ない場合は自動で既定値になります。")
        if pace_sel in PACE_NAMES and "hist" in S:
            lf = get_lifts(S["hist"]) if lift_mode.startswith("履歴") else {"win": PACE_DEFAULT_LIFT, "top3": PACE_DEFAULT_LIFT, "source": "定石の既定値"}
            with st.expander(f"倍率表（{lf['source']}）"):
                st.dataframe(pd.DataFrame({TACTICS_LABEL[t]: {f"{p_}": f"勝率×{lf['win'][p_][t]:.2f} / 複勝×{lf['top3'][p_][t]:.2f}" for p_ in PACE_NAMES} for t in (1, 2, 3, 4)}), **STRETCH)
        going = None if going_sel.startswith("入力どおり") else going_sel
        weather = None if weather_sel.startswith("入力どおり") else weather_sel
        with st.expander("出走馬の編集（出走取消・枠順・騎手・斤量・想定脚質）"):
            ed = st.data_editor(build_edit_table(race), hide_index=True, disabled=["Horse Name"], key=f"sc_{rid}_ed", **STRETCH,
                                column_config={"出走": st.column_config.CheckboxColumn("出走", help="外すと出走取消として扱います"),
                                               "想定脚質": st.column_config.SelectboxColumn("想定脚質", options=[TAC_AUTO] + list(TAC_CODE), help="ペース補正・展開分析にのみ使います"),
                                               "Bracket": st.column_config.NumberColumn("枠", min_value=1, max_value=8, step=1),
                                               "Horse Number": st.column_config.NumberColumn("馬番", min_value=1, max_value=18, step=1),
                                               "Burden Weight": st.column_config.NumberColumn("斤量", step=0.5)})
        spec_scope = "off"
        if bundle and (bundle.get("specialists") or {}).get("long"):
            sp_lab = st.radio("長距離専用モデルの適用範囲", ["菊花賞・天皇賞(春)・有馬記念のみ", "2500m以上の平地レースすべて", "使わない（汎用モデルのみ）"], horizontal=True, key="spec_scope")
            spec_scope = {"菊": "target", "2": "all_long"}.get(sp_lab[0], "off")
        clear_mode = "none"
        if not is_past:
            c1, c2 = st.columns([1, 1])
            save_on = c1.checkbox("予想と同時に predictions / race_results / race_management へ保存", value=True)
            clear_label = c2.radio("出走表バッファのリセット", ["予想したレースの行を削除（他レースは残す）", "出走表タブを全消去"], horizontal=False)
            clear_mode = "race" if clear_label.startswith("予想") else "all"
        else:
            save_on = st.checkbox("この結果で保存済みの予想を上書き保存する（predictions / race_results の AI予想列を更新）", value=False,
                                  help="オフのままなら、画面上の試算のみでスプレッドシートは変更しません。")
        b1, b2 = st.columns([3, 1])
        if b1.button("🔮 この条件で予想する", type="primary", disabled=bundle is None, key=f"run_{rid}"):
            try:
                with st.spinner("履歴参照 → 条件反映 → 特徴量生成 → 推論中…"):
                    hist, ped = get_hist()
                    res = run_scenario(bundle, hist, ped, race, spec_scope, going, weather, ed, pace_sel if pace_sel in PACE_NAMES else None, strength,
                                       get_lifts(hist) if lift_mode.startswith("履歴") else {"win": PACE_DEFAULT_LIFT, "top3": PACE_DEFAULT_LIFT, "source": "定石の既定値"})
                    scen = S["scen"].setdefault(rid, {})
                    lab = res["label"]; n_ = 2
                    while lab in scen:                                       # 同名シナリオは連番で区別
                        lab = f"{res['label']} #{n_}"; n_ += 1
                    res["label"] = lab; scen[lab] = res; S[f"scen_sel_{rid}"] = lab
                if save_on:
                    with st.spinner("スプレッドシートへ保存中（Upsert）…"):
                        save_prediction(sh, rid, res["race"], res["view"], res["pace"], dict(zip(res["view"]["Horse Name"], res["view"]["複勝率(%)"])), clear_mode)
                    res["saved"] = True
                    st.success("保存しました：predictions / race_results を更新、race_management を『予想済み』へ" + ("、出走表バッファをリセット" if clear_mode != "none" else ""))
            except Exception as exc:
                st.error(f"予想に失敗しました: {exc}"); st.code(traceback.format_exc())
        if b2.button("🧹 条件をリセット", key=f"reset_{rid}", **STRETCH):
            for k_ in [k for k in S.keys() if str(k).startswith(f"sc_{rid}_")]: del S[k_]
            st.rerun()

        scen = S["scen"].get(rid, {})
        if scen:
            labs = list(scen)
            if S.get(f"scen_sel_{rid}") not in labs: S[f"scen_sel_{rid}"] = labs[-1]
            st.subheader("3. AI予想結果")
            sel = st.selectbox("表示するシナリオ", labs, key=f"scen_sel_{rid}")
            P = scen[sel]
            st.caption(f"使用モデル: {P.get('model_label', '汎用モデル')}" + (f"／ペース補正: {P['pace_choice']}（倍率の出典: {P['lift_source']}・強さ{P['strength']}）" if P.get("pace_choice") else "")
                       + ("／保存済み" if P.get("saved") else "／未保存（画面上の試算）"))
            if P.get("missing_features"):
                st.warning(f"⚠ このモデルが使う特徴量のうち、アプリで生成できないものがあります: {P['missing_features']}（値0で代用して推論しているため精度が落ちます）。")
            if len(labs) >= 2:
                with st.expander(f"📊 シナリオ比較（{len(labs)}件）", expanded=True):
                    base_lab = st.selectbox("基準にするシナリオ", labs, key=f"cmp_base_{rid}")
                    cmp_df = compare_scenarios(scen, base_lab)
                    st.dataframe(cmp_df, hide_index=True, **STRETCH)
                    st.download_button("比較表CSV", to_csv_bytes(cmp_df), f"scenario_compare_{rid}.csv", "text/csv", key=f"cmp_dl_{rid}")
                    d1, d2 = st.columns(2)
                    if d1.button("🗑 表示中のシナリオを削除", key=f"del_{rid}"):
                        del S["scen"][rid][sel]; st.rerun()
                    if d2.button("🗑 全シナリオを削除", key=f"delall_{rid}"):
                        S["scen"][rid] = {}; st.rerun()
            P_race = P["race"]
            refresh = st.button("🔄 オッズ更新（odds_master）")
            with st.expander("💬 オッズ入力（ChatGPT等に問い合わせた結果の貼り付け／手入力）"):
                st.caption("① 下のプロンプトをAIに貼り付けて質問 → ② 返ってきたCSVを貼り付けて保存。AIの回答は誤りの可能性があるため、購入前にJRA公式等で必ず確認してください。")
                st.code(build_odds_prompt(day, label_txt, P_race), language="text")
                pasted = st.text_area("AIの回答（馬番,単勝,複勝下限,複勝上限）", height=140, key=f"odds_paste_{rid}")
                if pasted.strip():
                    pv = parse_odds_text(pasted); st.dataframe(pv, hide_index=True, **STRETCH)
                    if st.button("💾 貼り付けたオッズを保存", key=f"odds_save_p_{rid}"):
                        try:
                            ig = save_odds(sh, rid, P_race, pv, "AI問い合わせ貼り付け"); refresh = True
                            st.success("odds_master に保存しました" + (f"（出走表に無い馬番を無視: {ig}）" if ig else ""))
                        except Exception as exc: st.error(str(exc))
                cur_o = load_odds(sh, rid)
                base = pd.DataFrame({"Horse Number": P_race["Horse Number"].astype("Int64"), "Horse Name": P_race["Horse Name"]})
                if not cur_o.empty:
                    base = base.merge(cur_o.reindex(columns=["Horse Number", "Win Odds", "Place Odds Min", "Place Odds Max"]).assign(**{"Horse Number": lambda x: x["Horse Number"].astype("Int64")}), on="Horse Number", how="left")
                for c in ("Win Odds", "Place Odds Min", "Place Odds Max"):
                    if c not in base.columns: base[c] = np.nan
                st.caption("表で直接入力・修正もできます。")
                ed_o = st.data_editor(base, hide_index=True, disabled=["Horse Number", "Horse Name"], key=f"odds_ed_{rid}", **STRETCH)
                if st.button("💾 表のオッズを保存", key=f"odds_save_t_{rid}"):
                    try:
                        tv = ed_o.rename(columns={"Win Odds": "単勝", "Place Odds Min": "複勝下限", "Place Odds Max": "複勝上限"})
                        tv = tv.dropna(subset=["単勝", "複勝下限", "複勝上限"], how="all")
                        save_odds(sh, rid, P_race, tv[["Horse Number", "単勝", "複勝下限", "複勝上限"]], "手入力"); refresh = True
                        st.success("odds_master に保存しました")
                    except Exception as exc: st.error(str(exc))
            odds = load_odds(sh, rid, fresh=refresh)
            v = attach_odds(P["view"], odds)
            cols = ["予想順位", "Horse Number", "Horse Name", "勝率(%)"] + (["補正前勝率(%)"] if "補正前勝率(%)" in v.columns else []) + \
                   ["複勝率(%)", "単勝オッズ", "人気", "期待値(単勝)", "期待値(複勝)", "前走脚質", "想定脚質", "前走上がり3F", "前走着差(相対着順)", "間隔(日)"]
            st.dataframe(v[[c for c in cols if c in v.columns]], hide_index=True, **STRETCH)
            if odds.empty: st.caption("odds_master にこのレースのオッズがありません。期待値は表示されません。")
            else: st.caption("期待値 = 勝率 × 単勝オッズ（複勝は複勝率 × 複勝オッズ下限）。1.0超なら理論上プラス。オッズは学習に使わず、ここで動的結合しています。")

            st.subheader("4. 詳細レース考察")
            pc = P["pace"]
            if pc.get("ok"):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("想定ペース" + ("（指定）" if pc.get("user_override") else ""), pc["pace"]); m2.metric("ハイペース確率", f"{pc['high_prob']}%")
                m3.metric("逃げ・先行割合", f"{pc.get('front_ratio', 0)}%"); m4.metric("コース前残り率", f"{pc.get('course_front', 0)}%")
            st.bar_chart(pd.DataFrame({"頭数": [pc.get("n_nige", 0), pc.get("n_senko", 0), pc.get("n_sashi", 0), pc.get("n_oikomi", 0), pc.get("n_unknown", 0)]},
                                      index=["逃げ", "先行", "差し", "追込", "不明"]))
            for line in P["comments"]: st.markdown(line)
            b = P["bias"]
            if b.get("table") is not None:
                st.caption("同コース過去成績の枠別（内/中/外）"); st.dataframe(b["table"], **STRETCH)
            if b.get("ai") is not None:
                st.caption("今回のAI予測：枠別の平均勝率"); st.dataframe(b["ai"], **STRETCH)

            st.subheader("5. 買い目自動算出")
            bkey = (rid, sel)
            bc1, bc2 = st.columns(2)
            budget = bc1.number_input("予算（円）", min_value=100, step=100, value=3000, key=f"budget_{rid}")
            strat = bc2.selectbox("戦略", ["的中・堅実重視", "回収率・バリュー重視"], key=f"strat_{rid}")
            if st.button("🧠 買い目を自動算出する", key=f"bet_{rid}", type="primary"):
                S["bets"][bkey] = make_bets(v, int(budget), strat) + (int(budget), strat)
            if bkey in S["bets"]:
                bets, note, bud, st_name = S["bets"][bkey]
                if note: st.warning(note)
                if not bets: st.info("買い目を作成できませんでした。")
                else:
                    for bt in bets:
                        with st.container(border=True):
                            a, b_, c, d_ = st.columns([3, 1.4, 1.2, 1.6])
                            a.markdown(f"**{bt['種別']}　{bt['馬番']}番 {bt['馬名']}**")
                            b_.metric("金額", f"{bt['金額']:,}円"); c.metric("オッズ", f"{bt['オッズ']:.1f}倍")
                            d_.metric("的中確率 / 期待払戻", f"{bt['的中確率(%)']}% / {bt['期待払戻']:,}円")
                    tot = sum(x["金額"] for x in bets); er = sum(x["期待払戻"] for x in bets)
                    st.success(f"合計 {tot:,}円（予算 {bud:,}円・{st_name}・シナリオ: {sel}）／ 期待払戻 約{er:,}円（期待回収率 {er / tot * 100:.0f}%）")
                    st.caption("対象は odds_master にある単勝・複勝のみ。確率はAI推定で的中を保証するものではありません。馬券は自己責任で。")


# ------------------------------------------------------------------ タブ2
with tab2:
    st.subheader("予想 vs 実績 比較・検証")
    mg = load_race_management_df(sh, fresh=False)
    cand = mg[mg["Status"].isin([STATUS_PREDICTED, STATUS_RESULT])]
    with st.expander("📝 確定結果の登録（scrape_netkeiba.py のCSV／手入力）", expanded=cand.empty):
        st.caption("結果を登録すると race_results / predictions の Placing・Last 3F・Passing・脚質が更新され（AI予想列は保持）、race_management は『結果登録済み』になります。")
        up_r = st.file_uploader("結果CSV（23列。scrape_netkeiba.py の出力）", type=["csv"], key="res_csv")
        if up_r is not None:
            try:
                rdf = pd.read_csv(io.StringIO(decode_bytes(up_r.getvalue())), dtype=str, keep_default_na=False)
                st.caption(f"{len(rdf):,}行"); st.dataframe(standardize_to_23cols(rdf).head(20), hide_index=True, **STRETCH)
                if st.button("✅ CSVの結果を反映する", type="primary", key="apply_csv_res"):
                    with st.spinner("結果を反映中…"): r = apply_results(sh, rdf)
                    st.success(f"反映しました: {r}")
            except Exception as exc:
                st.error(f"結果の反映に失敗: {exc}"); st.code(traceback.format_exc())
        st.markdown("**手入力**（予想済みレースを選び、着順・上がり3F・通過順位を入力）")
        if not cand.empty:
            opts = {r["Race ID"]: f"{r['Race Day']} {r['Venue']} {r['Race Name']}（{r['Race ID']}）" for _, r in cand.iterrows()}
            mrid = st.selectbox("レース", list(opts), format_func=lambda x: opts[x], key="man_res_rid")
            rr0 = read_sheet_df(sh, RESULTS_SHEET)
            base_r = standardize_to_23cols(rr0[race_ids_vec(rr0) == mrid]) if not rr0.empty else pd.DataFrame()
            if base_r.empty:
                st.info("このレースの行が race_results にありません。")
            else:
                tbl = pd.DataFrame({"Horse Number": base_r["Horse Number"].astype("Int64"), "Horse Name": base_r["Horse Name"],
                                    "着順": base_r["Placing"].astype("Int64"), "上がり3F": base_r["Last 3F"], "通過順位": base_r["Passing"]})
                edr = st.data_editor(tbl, hide_index=True, disabled=["Horse Number", "Horse Name"], key=f"man_res_ed_{mrid}", **STRETCH)
                if st.button("✅ 手入力の結果を反映する", key="apply_man_res"):
                    try:
                        out = base_r.copy()
                        out["Placing"] = edr["着順"].values; out["Last 3F"] = edr["上がり3F"].values; out["Passing"] = edr["通過順位"].fillna("").values
                        r = apply_results(sh, out); st.success(f"反映しました: {r}")
                    except Exception as exc: st.error(str(exc))
    if cand.empty:
        st.info("予想済みのレースがありません。")
    else:
        vday = st.selectbox("検証する開催日", sorted(cand["Race Day"].unique(), reverse=True), key="verify_day")
        if st.button("🔍 検証開始（予想と確定結果を照合）", type="primary"):
            rids = list(cand.loc[cand["Race Day"] == vday, "Race ID"])
            rr = read_sheet_df(sh, RESULTS_SHEET, fresh=True)
            sub = rr[race_ids_vec(rr).isin(rids)] if not rr.empty else rr
            ok, problems = check_23_completeness(sub, rids)
            if not ok:
                st.markdown("<div style='border:3px solid #d00;padding:14px;border-radius:8px;background:#fff0f0'>"
                            "<h3 style='color:#d00;margin:0'>⚠️ 取得した結果データが不完全なため、23列の整合性を満たしていません。</h3>"
                            "<p style='color:#d00;font-weight:bold;margin:6px 0 0'>正しい結果が反映されるまで検証を中断します。</p></div>", unsafe_allow_html=True)
                with st.expander("不足・欠損の内訳", expanded=True):
                    for p in problems: st.write("・" + p)
            else:
                vdf = build_verification(sub, read_sheet_df(sh, ODDS_SHEET))
                n = len(vdf)
                m1, m2, m3 = st.columns(3)
                m1.metric("対象レース", f"{n}"); m2.metric("本命 単勝的中率", f"{vdf['単勝的中'].mean() * 100:.1f}%")
                m3.metric("本命 複勝的中率", f"{vdf['複勝的中'].mean() * 100:.1f}%")
                w = vdf.dropna(subset=["単勝払戻(100円)"]); p_ = vdf.dropna(subset=["複勝払戻(100円)"])
                m4, m5, m6 = st.columns(3)
                m4.metric("単勝回収率（本命100円）", f"{w['単勝払戻(100円)'].sum() / (100 * len(w)) * 100:.0f}%" if len(w) else "オッズ無し")
                m5.metric("複勝回収率（本命・下限）", f"{p_['複勝払戻(100円)'].sum() / (100 * len(p_)) * 100:.0f}%" if len(p_) else "オッズ無し")
                m6.metric("AI上位3頭中の3着内（平均）", f"{vdf['AI上位3頭中の3着内'].mean():.2f}頭")
                st.dataframe(vdf, hide_index=True, **STRETCH)
                st.bar_chart(pd.DataFrame({"率(%)": [vdf["単勝的中"].mean() * 100, vdf["複勝的中"].mean() * 100]}, index=["単勝的中率", "複勝的中率"]))
                if len(w):
                    cum = (w["単勝払戻(100円)"] - 100).cumsum().reset_index(drop=True)
                    st.line_chart(pd.DataFrame({"累積収支(円・単勝本命100円)": cum}))
                st.download_button("検証結果CSV", to_csv_bytes(vdf), "verification.csv", "text/csv")

# ------------------------------------------------------------------ タブ3
with tab3:
    st.subheader("A. 出走表の手入力")
    with st.expander("手入力フォームを開く"):
        m1, m2, m3, m4 = st.columns([1.2, 1, 0.8, 2])
        me_day = m1.date_input("開催日", value=datetime.now(JST).date(), key="me_day"); me_venue = m2.selectbox("競馬場", VENUES, index=5, key="me_venue")
        me_rno = m3.number_input("R", 1, 12, 1, key="me_rno"); me_name = m4.text_input("レース名（空欄なら『会場+R』）", key="me_name")
        n1, n2, n3, n4, n5 = st.columns([1, 1, 1, 1, 2])
        me_dist = n1.number_input("距離(m)", 1000, 3600, 1600, step=100, key="me_dist"); me_tt = n2.selectbox("芝/ダ", ["芝", "ダ", "障"], key="me_tt")
        me_going = n3.selectbox("馬場", ["良", "稍重", "重", "不良"], key="me_going"); me_weather = n4.selectbox("天候", ["晴", "曇", "小雨", "雨", "小雪", "雪"], key="me_weather")
        me_key = n5.text_input("レースID（12桁・任意。結果CSVとの突合に使用）", key="me_key")
        base_e = pd.DataFrame({"Bracket": pd.array([None] * 18, dtype="Int64"), "Horse Number": list(range(1, 19)), "Horse Name": [""] * 18, "性齢": [""] * 18,
                               "Burden Weight": [None] * 18, "Jockey": [""] * 18, "Trainer": [""] * 18})
        ed_e = st.data_editor(base_e, num_rows="dynamic", hide_index=True, key="me_editor", **STRETCH)
        if st.button("出走表タブへ保存（Race ID単位でUpsert）", key="me_save"):
            ent_in = ed_e[ed_e["Horse Name"].astype(str).str.strip() != ""].copy()
            if ent_in.empty: st.error("馬名が1頭も入力されていません。")
            else:
                ent_in["Race Day"] = me_day.strftime("%Y-%m-%d"); ent_in["Venue"] = me_venue
                ent_in["Race Name"] = me_name.strip() or f"{me_venue}{int(me_rno)}R"
                ent_in["Distance"] = me_dist; ent_in["Track_Type"] = me_tt; ent_in["Track Condition"] = me_going; ent_in["Weather"] = me_weather
                ent_in["Race_Key_Src"] = me_key.strip()
                r = upsert_by_race(sh, ENTRY_SHEET, standardize_to_23cols(ent_in), COLUMNS_23); st.success(f"保存しました: {r}")

    st.subheader("B. CSV一括同期（ファイルアップロード → 指定タブ）")
    st.caption("確定結果のCSVは、タブ②の「確定結果の登録」からアップロードすると AI予想列を保持したまま反映できます。")
    targets = [ENTRY_SHEET, PREDICTIONS_SHEET, RESULTS_SHEET, ODDS_SHEET, HORSES_SHEET, RACE_MANAGEMENT_SHEET]
    tgt = st.selectbox("書き込み先タブ", targets, index=1)
    mode_label = st.radio("書き込み方法", ["Upsert（Race ID単位で更新・追加／血統は馬名単位）", "追記", "タブ全体を上書き"], horizontal=True)
    mode = {"U": "upsert", "追": "append", "タ": "replace"}[mode_label[0]]
    std = st.checkbox("23列へ自動標準化（standardize_to_23cols）", value=tgt in (ENTRY_SHEET, PREDICTIONS_SHEET, RESULTS_SHEET), disabled=tgt not in (ENTRY_SHEET, PREDICTIONS_SHEET, RESULTS_SHEET))
    up = st.file_uploader("CSVファイル", type=["csv"], key="csv_sync")
    if up is not None:
        try:
            prev = pd.read_csv(io.StringIO(decode_bytes(up.getvalue())), dtype=str, keep_default_na=False)
            prev_view = standardize_to_23cols(prev) if std else prev
            st.caption(f"{len(prev):,}行 / {prev_view.shape[1]}列"); st.dataframe(prev_view.head(30), hide_index=True, **STRETCH)
            if tgt == HORSES_SHEET and (prev_view.get("Horse Name", pd.Series(dtype=str)).astype(str).str.strip() == "").all():
                st.error("Horse Name がすべて空欄です。馬名で結合できないため血統特徴に使えません。")
            if mode == "replace": st.warning(f"「{tgt}」タブの内容を全て置き換えます。")
            if st.button(f"🚀 「{tgt}」へ同期する", type="primary"):
                with st.spinner("書き込み中…"):
                    res = upload_csv_to_spreadsheet(up, tgt, mode=mode, standardize=std, sh=sh)
                st.success(f"同期完了: {res}")
        except Exception as exc:
            st.error(f"同期に失敗: {exc}"); st.code(traceback.format_exc())

    st.subheader("C. 出走表の貼り付け入力（TSV/CSV）")
    txt = st.text_area("出走表テキスト（列名付き。性齢『牡3』も自動分解）", height=150)
    if txt.strip():
        try:
            std_df = standardize_to_23cols(parse_input_table(txt)); st.dataframe(std_df, hide_index=True, **STRETCH)
            if st.button("出走表タブへ書き込む（Race ID単位でUpsert）"):
                r = upsert_by_race(sh, ENTRY_SHEET, std_df, COLUMNS_23); st.success(f"書き込み完了: {r}")
        except Exception as exc: st.error(f"解析に失敗: {exc}")

# ------------------------------------------------------------------ タブ4
with tab4:
    st.subheader("レース管理（Race IDユニーク）")
    mg = load_race_management_df(sh, fresh=st.button("🔄 再読込", key="mg_reload"))
    if mg.empty:
        st.info("race_management は空です。出走表を読み込むと『未予想』で自動登録されます。")
    else:
        cnt = mg["Status"].value_counts()
        a, b, c = st.columns(3)
        a.metric(STATUS_UNPREDICTED, int(cnt.get(STATUS_UNPREDICTED, 0))); b.metric(STATUS_PREDICTED, int(cnt.get(STATUS_PREDICTED, 0)))
        c.metric(STATUS_RESULT, int(cnt.get(STATUS_RESULT, 0)))
        stf = st.multiselect("ステータス絞り込み", list(STATUS_ORDER), default=list(STATUS_ORDER))
        st.dataframe(mg[mg["Status"].isin(stf)].sort_values("Race ID", ascending=False), hide_index=True, **STRETCH)
        with st.expander("ステータスを手動で変更"):
            rid_sel = st.selectbox("Race ID", list(mg["Race ID"]), key="mg_rid")
            new_st = st.selectbox("新しいステータス", list(STATUS_ORDER), key="mg_status")
            if st.button("変更する"):
                row = mg[mg["Race ID"] == rid_sel].iloc[0]
                ws, hd = ensure_sheet(sh, RACE_MANAGEMENT_SHEET, MANAGEMENT_COLUMNS)
                cell = ws.find(str(rid_sel), in_column=1)
                ws.update_cell(cell.row, hd.index("Status") + 1, new_st)
                ws.update_cell(cell.row, hd.index("Updated At") + 1, now_str())
                clear_sheet_cache(); st.success("更新しました（再読込で反映）")
        st.download_button("race_management CSV", to_csv_bytes(mg), "race_management.csv", "text/csv")

# ------------------------------------------------------------------ タブ5
with tab5:
    st.subheader("血統マスタ（horses_master：23列とは独立）")
    hm = read_sheet_df(sh, HORSES_SHEET, fresh=st.button("🔄 再読込", key="hm_reload"))
    if hm.empty:
        st.info("horses_master は空です。下のフォームかタブ③のCSV同期で登録してください。")
    else:
        hm = hm.reindex(columns=HORSES_COLUMNS).fillna("")
        blank = int((hm["Horse Name"].str.strip() == "").sum())
        if blank: st.error(f"馬名が空欄の行が {blank} 件あります。この行は血統特徴として使われません。")
        kw = st.text_input("馬名・父・母父で検索", key="hm_kw")
        view_h = hm[hm.apply(lambda r: kw in " ".join(r.astype(str)), axis=1)] if kw else hm
        st.caption(f"{len(view_h):,} / {len(hm):,} 頭"); st.dataframe(view_h, hide_index=True, **STRETCH)
        st.download_button("horses_master CSV", to_csv_bytes(hm), "horses_master.csv", "text/csv")
    with st.form("horse_form", clear_on_submit=True):
        f1, f2, f3 = st.columns(3)
        hn = f1.text_input("馬名"); hs = f2.text_input("父 (Sire)"); hd_ = f3.text_input("母父 (Dam's Sire)")
        if st.form_submit_button("登録・更新（馬名で Upsert）"):
            if not hn.strip(): st.error("馬名は必須です。")
            else:
                upsert_horses(sh, pd.DataFrame([{"Horse Name": hn.strip(), "Sire": hs.strip(), "Dam's Sire": hd_.strip()}]))
                st.success(f"{hn} を登録しました。サイドバーの『履歴・血統を再読込』で反映されます。")
