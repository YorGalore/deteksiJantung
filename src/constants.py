"""Konstanta bersama proyek deteksi AMI.

Satu-satunya sumber aturan: kode ICD, daftar kolom, sentinel, rentang valid,
dan pola kata kunci. Orang 1, 2, dan 3 memakai file ini, bukan menulis ulang
daftar sendiri. Setiap perubahan dicatat di reports/decision_log.md.
"""

# ---------------------------------------------------------------------------
# Kohort
# ---------------------------------------------------------------------------
FOLD_COL = "fold"            # kolom split resmi Ext-ICD (bukan strat_fold)
FOLDS_BASE_TRAIN = list(range(0, 17))   # 0-16: model dasar Jalur A
FOLD_META = 17                          # meta-learner
FOLD_THRESHOLD = 18                     # kalibrasi & threshold
FOLD_TEST = 19                          # test akhir (dikunci)

# ---------------------------------------------------------------------------
# Label (kode ICD-10 tanpa titik, huruf besar)
# Kode yang ditemukan di data (eksplorasi 28-09-2026):
#   I2101 I2102 I2109 I2111 I2119 I2121 I2129 I213 I214 I219
#   I21A1 I21A9 I220 I221 I222 I228 I229
# ---------------------------------------------------------------------------
AMI_PREFIXES = ("I21", "I22")
STEMI_PREFIXES = ("I210", "I211", "I212", "I213")
NSTEMI_PREFIXES = ("I214",)
AMI_UNSPECIFIED_PREFIXES = ("I219",)
AMI_TYPE2_PREFIXES = ("I21A1",)
AMI_OTHER_PREFIXES = ("I21A9",)
AMI_SUBSEQUENT_PREFIXES = ("I22",)

# ---------------------------------------------------------------------------
# Machine measurements
# ---------------------------------------------------------------------------
MM_SENTINEL_MIN = 29999      # nilai >= ini dianggap kosong (sentinel)

# Kolom mentah yang dibaca. Nilai onset/end mentah TIDAK dijadikan fitur karena
# titik acuannya bergantung pada jenis mesin (p_onset=40 konstan, qrs_onset ~200
# vs ~500); yang dipakai hanya selisihnya.
MM_RAW_COLS = ["rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end",
               "t_end", "p_axis", "qrs_axis", "t_axis"]

# Rentang valid (di luar rentang -> NaN)
MM_VALID_RANGE = {
    "rr_interval": (240, 3000),
    "hr_ecg": (20, 250),
    "pr_interval": (60, 400),
    "qrs_duration": (40, 250),
    "qt_interval": (200, 700),
    "qtc_bazett": (250, 700),
    "p_axis": (-180, 360),
    "qrs_axis": (-180, 360),
    "t_axis": (-180, 360),
}

# ---------------------------------------------------------------------------
# Triase
# ---------------------------------------------------------------------------
TRIAGE_VALID_RANGE = {
    "temperature": (90, 110),   # Fahrenheit
    "heartrate": (20, 250),
    "resprate": (4, 60),
    "o2sat": (50, 100),
    "sbp": (50, 300),
    "dbp": (20, 200),
    "pain": (0, 10),
    "acuity": (1, 5),
}

# ---------------------------------------------------------------------------
# Keluhan utama (regex, dicocokkan pada teks huruf kecil)
# ---------------------------------------------------------------------------
CC_PATTERNS = {
    "cc_chest_pain": r"\b(chest pain|cp|chest pressure|chest tightness|chest discomfort)\b",
    "cc_dyspnea": r"\b(sob|dyspnea|shortness of breath|difficulty breathing)\b",
    "cc_syncope": r"\b(syncope|syncopal|passed out|loc)\b",
    "cc_weakness": r"\b(weakness|fatigue|malaise)\b",
    "cc_epigastric_abd": r"\b(epigastric|abd pain|abdominal pain)\b",
    "cc_palpitations": r"\b(palpitations|palps|racing heart)\b",
    "cc_dizziness": r"\b(dizzy|dizziness|lightheaded)\b",
    "cc_nausea_vomit": r"\b(nausea|vomiting|n/v)\b",
    "cc_diaphoresis": r"\b(diaphoresis|diaphoretic|sweating)\b",
    "cc_arm_jaw_pain": r"\b(arm pain|jaw pain)\b",
    "cc_back_pain": r"\bback pain\b",
}
# Hanya untuk analisis sensitivitas, BUKAN fitur
CC_EXPLICIT_MI_PATTERN = r"\b(stemi|nstemi|mi|acs|code stemi|cath|myocardial infarction)\b"

# ---------------------------------------------------------------------------
# Golongan obat dari medrecon (kolom `name`, huruf kecil)
# ---------------------------------------------------------------------------
MED_PATTERNS = {
    "med_antiplatelet": r"aspirin|clopidogrel|ticagrelor|prasugrel",
    "med_anticoagulant": r"warfarin|apixaban|rivaroxaban|dabigatran|enoxaparin",
    "med_statin": r"atorvastatin|simvastatin|rosuvastatin|pravastatin|lovastatin",
    "med_betablocker": r"metoprolol|atenolol|carvedilol|propranolol|labetalol|bisoprolol|nadolol",
    "med_acei_arb": r"lisinopril|enalapril|ramipril|captopril|benazepril|losartan|valsartan|irbesartan|olmesartan|candesartan",
    "med_nitrate": r"nitroglycerin|isosorbide",
    "med_insulin": r"insulin",
    "med_oral_antidiabetic": r"metformin|glipizide|glyburide|glimepiride|sitagliptin|pioglitazone",
    "med_loop_diuretic": r"furosemide|torsemide|bumetanide",
    "med_ccb": r"amlodipine|diltiazem|verapamil|nifedipine",
}

# ---------------------------------------------------------------------------
# Kelompok kolom tabel master
# ---------------------------------------------------------------------------
ID_COLS = ["ed_stay_id", "subject_id", "study_id", "file_name"]
SPLIT_COLS = [FOLD_COL, "internal_val_0_16", "internal_val_0_17"]
LABEL_COLS = ["label_ami", "label_source", "ami_stemi", "ami_nstemi",
              "ami_unspecified", "ami_type2", "ami_other", "ami_subsequent",
              "ami_type2_only"]
ANALYSIS_COLS = ["cc_explicit_mi", "cc_missing", "medrecon_available",
                 "mm_missing_count", "first_ecg_matches_no0", "signal_ok"]

EMR_FEATURES = (
    ["age", "is_male", "arrival_transport",
     "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
     "pain", "acuity"]
    + list(CC_PATTERNS)
    + list(MED_PATTERNS)
    + ["n_meds"]
)
MM_FEATURES = ["rr_interval", "hr_ecg", "pr_interval", "qrs_duration",
               "qt_interval", "qtc_bazett", "p_axis", "qrs_axis", "t_axis",
               "p_wave_absent"]

# Kolom yang tidak boleh pernah masuk matriks fitur
FORBIDDEN_COLUMNS = (
    [f"report_{i}" for i in range(18)]
    + ["cart_id", "bandwidth", "filtering", "ecg_time",
       "p_onset", "p_end", "qrs_onset", "qrs_end", "t_end",
       "ed_diag_ed", "ed_diag_hosp", "hosp_diag_hosp", "all_diag_hosp", "all_diag_all",
       "ed_hadm_id", "hosp_hadm_id", "hadm_id", "dod", "disposition",
       "intime", "outtime", "race", "chiefcomplaint",
       "ecg_no_within_stay", "ecg_taken_in_hosp", "ecg_taken_in_ed_or_hosp",
       "strat_fold", "anchor_year", "anchor_age"]
    + LABEL_COLS + ANALYSIS_COLS + SPLIT_COLS
)
