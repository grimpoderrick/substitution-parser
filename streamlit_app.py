import io
import re
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
SAMPLE_DIR = BASE_DIR / "sample_files"

# Try logo.png in root, then in /logo
LOGO_PATH = None
for cand in [BASE_DIR / "logo.png", BASE_DIR / "logo" / "logo.png"]:
    if cand.exists():
        LOGO_PATH = cand
        break

# --- PAGE SETUP ----------------------------------------------------------------
st.set_page_config(
    page_title="Substitution Parser",
    page_icon="🔁",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
SAMPLE_DIR = BASE_DIR / "sample_files"

# Try logo.png in root, then in /logo
LOGO_PATH = None
for cand in [BASE_DIR / "logo.png", BASE_DIR / "logo" / "logo.png"]:
    if cand.exists():
        LOGO_PATH = cand
        break

# --- STYLING -------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1100px;
        padding-top: 0.8rem;
        padding-bottom: 1.5rem;
    }
    .big-title {
        font-size: 2.4rem;
        font-weight: 700;
        text-align: left;
        margin-bottom: 0.1rem;
    }
    .subheader-text {
        color: #4b5563;
        text-align: left;
        font-size: 0.96rem;
        margin-bottom: 0.9rem;
    }
    .section-card {
        background-color: #f5f9ff;
        padding: 0.9rem 1.1rem;
        border-radius: 0.75rem;
        margin-bottom: 0.8rem;
        border: 1px solid #e0e7ff;
    }
    .section-title {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        border-radius: 999px;
        background: #2563eb;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
    }
    /* NEW: smaller example buttons */
    .example-btn-wrapper button {
        font-size: 0.75rem;
        padding: 0.15rem 0.6rem;
        min-width: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- HELPERS -------------------------------------------------------------------
def find_first_existing_file(folder: Path, names):
    """Return first existing Path in folder from names list, or None."""
    for n in names:
        p = folder / n
        if p.exists():
            return p
    return None


def read_flat_headerless_tab(uploaded):
    """Read a headerless tab-delimited flat file with RECORD | question | value."""
    if uploaded is None:
        return None

    df = pd.read_csv(
        uploaded,
        sep="\t",
        header=None,
        names=["RECORD", "question", "value"],
        dtype=str,
        encoding="utf-8",
        engine="python",
    )

    if df.shape[1] != 3:
        raise ValueError(
            f"Expected 3 columns (RECORD, question, value); found {df.shape[1]}."
        )

    return df


def read_attribute_table(uploaded):
    """Read attribute table (first col = ID, second = Description, rest = attributes)."""
    suffix = Path(uploaded.name).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded)
    elif suffix in [".csv"]:
        return pd.read_csv(uploaded)
    else:  # txt / tsv
        return pd.read_csv(uploaded, sep="\t")

def parse_flat_all_pairs_local(
    df_long: pd.DataFrame,
    id_col: str = "RECORD",
    question_col: str = "question",
    value_col: str = "value",
    multisub_weighting: str = "equal_share",
) -> pd.DataFrame:
    """
    Local copy of parse_flat_all_pairs from flat_loader.py.

    Input: long flat file with RECORD, question, value.
    Uses HidProduct{instance} as ORIGINAL and X1_Lr{instance}r{code} as SUBSTITUTE.
    Includes 9998/9999 "no-substitute" codes.
    Returns columns: Record, Instance, Original, Substitute
    """
    w = df_long[[id_col, question_col, value_col]].copy()
    w[question_col] = w[question_col].astype(str).str.strip()
    w[value_col] = w[value_col].astype(str).str.strip()

    # Originals
    orig = {}
    hid_mask = w[question_col].str.match(r"(?i)^HidProduct(\d+)$")
    for rec, q, val in w.loc[hid_mask, [id_col, question_col, value_col]].itertuples(index=False):
        if pd.isna(val) or str(val).strip() == "":
            continue
        inst = int(re.match(r"(?i)^HidProduct(\d+)$", q).group(1))
        if str(val).replace(".", "", 1).isdigit():
            orig[(rec, inst)] = str(int(float(val)))
        else:
            orig[(rec, inst)] = str(val)

    # Substitutes (include no-sub)
    subs_by_key = {}
    sel_tokens = {"1", "1.0", "yes", "true", "selected", "y"}
    val_norm = w[value_col].astype(str).str.strip().str.lower()
    val_is_selected = val_norm.isin(sel_tokens) | (
        pd.to_numeric(w[value_col], errors="coerce").fillna(0) == 1
    )
    bin_mask = w[question_col].str.match(r"(?i)^X1_Lr(\d+)r(\d+)$") & val_is_selected

    for rec, q, _ in w.loc[bin_mask, [id_col, question_col, value_col]].itertuples(index=False):
        m = re.match(r"(?i)^X1_Lr(\d+)r(\d+)$", q)
        inst = int(m.group(1))
        code = m.group(2)  # keep 9998/9999/etc
        subs_by_key.setdefault((rec, inst), []).append(str(code))

    rows = []
    for (rec, inst), subs in subs_by_key.items():
        if (rec, inst) not in orig:
            continue
        if multisub_weighting == "first_only":
            subs = subs[:1]
        for sub in subs:
            rows.append((rec, inst, orig[(rec, inst)], sub))

    out = pd.DataFrame(rows, columns=["Record", "Instance", "Original", "Substitute"])
    return out.sort_values(
        ["Record", "Instance", "Original", "Substitute"]
    ).reset_index(drop=True)


def build_substitution_pairs_from_flat(flat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either:
      • raw flat file: columns [RECORD, question, value]  → parse via HidProduct/X1_Lr
      • already-parsed pairs file: columns like Record/Instance/Original/Substitute
        or RECORD/INSTANCE/Original/Substitute, etc.

    Returns DataFrame with columns exactly: Record, Instance, Original, Substitute.
    """
    cols = list(flat_df.columns)
    lower = {c.lower() for c in cols}

    # Case 1: raw headerless flat (what you're uploading as .txt)
    if lower == {"record", "question", "value"}:
        # ensure expected names
        df_long = flat_df.rename(columns={cols[0]: "RECORD", cols[1]: "question", cols[2]: "value"})
        return parse_flat_all_pairs_local(df_long)

    # Case 2: already-parsed pairs (Substitution Example-style)
    if {"record", "instance", "original", "substitute"}.issubset(lower):
        mapping = {}
        for c in cols:
            cl = c.lower()
            if cl == "record":
                mapping[c] = "Record"
            elif cl == "instance":
                mapping[c] = "Instance"
            elif cl in ("original", "selectedproductcode", "product_code_original"):
                mapping[c] = "Original"
            elif cl in ("substitute", "productcode", "product_code_sub"):
                mapping[c] = "Substitute"
        df_pairs = flat_df.rename(columns=mapping)
        return df_pairs[["Record", "Instance", "Original", "Substitute"]]

    raise ValueError(
        "Unrecognized flat file structure. Expected either:\n"
        " • headerless flat with columns [RECORD, question, value], or\n"
        " • pairs file with Record / Instance / Original / Substitute.\n"
        f"Got columns: {cols}"
    )


def combine_substitution_and_attributes(pairs_df: pd.DataFrame, attr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine substitution pairs with attribute table.

    pairs_df columns: Record, Instance, Original, Substitute
    Attribute table: first col = ID, second = Description, rest = attributes.

    Output column order:
      RECORD, INSTANCE, OriginalProductCode, SubstitutedProductCode,
      Orig_Description, Orig_<other attrs...>, Sub_Description, Sub_<other attrs...>
    """
    df = pairs_df.copy()
    required = {"Record", "Instance", "Original", "Substitute"}
    if not required.issubset(df.columns):
        raise ValueError(f"pairs_df must contain columns {required}, got {df.columns.tolist()}")

    df["RECORD"] = df["Record"]
    df["INSTANCE"] = df["Instance"]
    df["OriginalProductCode"] = df["Original"].astype(str)
    df["SubstitutedProductCode"] = df["Substitute"].astype(str)

    attr = attr_df.copy()
    if attr.shape[1] < 2:
        raise ValueError("Attribute file must have at least two columns (ID and Description).")

    id_col = attr.columns[0]
    desc_col = attr.columns[1]
    attr[id_col] = attr[id_col].astype(str)

    orig_attr = attr.rename(columns=lambda c: f"Orig_{c}")
    sub_attr = attr.rename(columns=lambda c: f"Sub_{c}")

    merged = df.merge(
        orig_attr,
        left_on="OriginalProductCode",
        right_on=f"Orig_{id_col}",
        how="left",
    )
    merged = merged.merge(
        sub_attr,
        left_on="SubstitutedProductCode",
        right_on=f"Sub_{id_col}",
        how="left",
    )

    base_cols = ["RECORD", "INSTANCE", "OriginalProductCode", "SubstitutedProductCode"]

    # Original attributes: description first
    orig_all = [c for c in merged.columns if c.startswith("Orig_")]
    id_orig = f"Orig_{id_col}"
    if id_orig in orig_all:
        orig_all.remove(id_orig)
    desc_orig = f"Orig_{desc_col}"
    if desc_orig in orig_all:
        orig_all.remove(desc_orig)
        orig_cols = [desc_orig] + orig_all
    else:
        orig_cols = orig_all

    # Substituted attributes: description first
    sub_all = [c for c in merged.columns if c.startswith("Sub_")]
    id_sub = f"Sub_{id_col}"
    if id_sub in sub_all:
        sub_all.remove(id_sub)
    desc_sub = f"Sub_{desc_col}"
    if desc_sub in sub_all:
        sub_all.remove(desc_sub)
        sub_cols = [desc_sub] + sub_all
    else:
        sub_cols = sub_all

    # Make ID / code fields numeric where possible
    for col in ["RECORD", "INSTANCE", "OriginalProductCode", "SubstitutedProductCode"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="ignore")

    ordered = [c for c in base_cols + orig_cols + sub_cols if c in merged.columns]
    return merged[ordered]

# --- HEADER --------------------------------------------------------------------
col_logo, col_title = st.columns([1, 4])

with col_logo:
    if LOGO_PATH:
        # bigger logo
        st.image(str(LOGO_PATH), width=240)

with col_title:
    st.markdown('<div class="big-title">Substitution Parser</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subheader-text">'
        'Upload your <strong>flat substitution file</strong> and '
        '<strong>attribute table</strong>, and get a parsed file with attributes '
        'for the original and substituted products.'
        '</div>',
        unsafe_allow_html=True,
    )

# ---- Need example files bar ---------------------------------------------------
flat_sample = find_first_existing_file(
    SAMPLE_DIR,
    ["Flat File Example.txt", "Substitution Example.txt", "Substitution Example.xlsx"],
)
attr_sample = find_first_existing_file(
    SAMPLE_DIR,
    ["Attribute Table Example.xlsx", "Attributes Example.xlsx"],
)

# 5 columns: left spacer, text, button1, button2, right spacer
spacer_left, col_text, col_flat_btn, col_attr_btn, spacer_right = st.columns(
    [2, 1, 1.4, 1.6, 2]
)

with col_text:
    st.markdown(
        "<div style='text-align:center; font-weight:600;'>Need example files?</div>",
        unsafe_allow_html=True,
    )

with col_flat_btn:
    if flat_sample:
        with open(flat_sample, "rb") as f:
            st.download_button(
                "Download Example Flat File",
                data=f,
                file_name=flat_sample.name,
                mime=(
                    "text/plain"
                    if flat_sample.suffix == ".txt"
                    else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                ),
                key="flat_example_btn_top",
            )

with col_attr_btn:
    if attr_sample:
        with open(attr_sample, "rb") as f:
            st.download_button(
                "Download Example Attribute File",
                data=f,
                file_name=attr_sample.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="attr_example_btn_top",
            )

st.markdown("---")




# --- SAMPLE FILE PATHS --------------------------------------------------------
flat_sample = find_first_existing_file(
    SAMPLE_DIR,
    ["Flat File Example.txt", "Substitution Example.xlsx", "Substitution Example.txt"],
)
attr_sample = find_first_existing_file(
    SAMPLE_DIR,
    ["Attribute Table Example.xlsx", "Attributes Example.xlsx"],
)

# --- STEP 1: UPLOAD FLAT FILE -------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)

st.markdown(
    '<div class="section-title"><span class="step-badge">1</span>'
    'Upload Flat File</div>',
    unsafe_allow_html=True,
)

flat_file = st.file_uploader(
    "Upload Flat Substitution File",
    type=["txt", "tsv", "xlsx", "xls", "csv"],
    key="flat_file",
    help=(
        "Either:\n"
        " • headerless, tab-delimited with columns RECORD | question | value, or\n"
        " • an already-parsed pairs file like the Substitution Example."
    ),
)
st.markdown("</div>", unsafe_allow_html=True)


flat_df = None
if flat_file is not None:
    try:
        # If it's a text/tsv, assume headerless flat; otherwise, let pandas read with header.
        suffix = Path(flat_file.name).suffix.lower()
        if suffix in [".txt", ".tsv"]:
            flat_df = read_flat_headerless_tab(flat_file)
        elif suffix in [".xlsx", ".xls"]:
            flat_df = pd.read_excel(flat_file)
        else:  # csv
            flat_df = pd.read_csv(flat_file)

        st.success(
            f"Flat file loaded. Shape: {flat_df.shape[0]:,} rows × {flat_df.shape[1]} columns."
        )
        st.caption(f"Columns: {list(flat_df.columns)}")
    except Exception as e:
        st.error(f"Could not read flat file: {e}")

# --- STEP 2: UPLOAD ATTRIBUTE TABLE ------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)

col_step2, col_btn2, col_spacer2 = st.columns([2, 1, 8])

attr_file = st.file_uploader(
    "Upload Attribute Table",
    type=["xlsx", "xls", "csv", "tsv", "txt"],
    key="attr_file",
)
st.markdown("</div>", unsafe_allow_html=True)

attr_df = None
if attr_file is not None:
    try:
        attr_df = read_attribute_table(attr_file)
        st.success(
            f"Attribute file loaded. Shape: {attr_df.shape[0]:,} rows × {attr_df.shape[1]} columns."
        )
    except Exception as e:
        st.error(f"Could not read attribute file: {e}")

# --- STEP 3: PARSE & COMBINE --------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title"><span class="step-badge">3</span>'
    'Pull Substitution and Combine with Attributes</div>',
    unsafe_allow_html=True,
)

if flat_df is not None and attr_df is not None:
    if st.button("Pull Substitution and Combine with Attributes", type="primary"):
        try:
            # 1) Build original→substitution pairs from flat
            pairs_df = build_substitution_pairs_from_flat(flat_df)

            # 2) Combine with attribute table
            combined = combine_substitution_and_attributes(pairs_df, attr_df)

            n_pairs = len(pairs_df)
            n_combined = len(combined)

            m1, m2 = st.columns(2)
            m1.metric("Rows in parsed substitution pairs", f"{n_pairs:,}")
            m2.metric("Rows after attribute join", f"{n_combined:,}")

            st.markdown("#### Preview")
            st.dataframe(combined.head(100), use_container_width=True)

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                combined.to_excel(writer, index=False, sheet_name="Substitution Parsed")
            st.download_button(
                "Download Parsed File (.xlsx)",
                data=buffer.getvalue(),
                file_name="Substitution_Parsed.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            st.error(f"Something went wrong while parsing: {e}")
else:
    st.info("Upload both the flat substitution file and the attribute table to enable the parser.")

st.markdown("</div>", unsafe_allow_html=True)
