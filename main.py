import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO, BytesIO
from typing import Dict, Tuple, Any

# =========================
# Streamlit page configuration
# =========================
st.set_page_config(page_title='CSV Analyzer', layout='wide', page_icon='📊')

# =========================
# Helper Functions
# =========================
@st.cache_data
def read_csv_file(f) -> pd.DataFrame:
    """Reads CSV from file or string-like object robustly"""
    try: f.seek(0)
    except Exception: pass
    raw = f.read() if hasattr(f, 'read') else str(f)
    text = raw.decode('utf-8', errors='replace') if isinstance(raw, (bytes, bytearray)) else str(raw)
    for engine in (None, ',', ';', '\t', '|'):
        try:
            return pd.read_csv(StringIO(text), sep=engine, engine='python' if engine is None else 'c')
        except Exception:
            continue
    raise ValueError('Could not parse CSV')

def df_to_bytes(df: pd.DataFrame, fmt: str = 'csv') -> bytes:
    """Convert DataFrame to bytes for download"""
    if fmt == 'csv':
        return df.to_csv(index=False).encode('utf-8')
    b = BytesIO()
    df.to_excel(b, index=False)
    return b.getvalue()

@st.cache_data
def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Generate concise profile per column: dtype, missing, unique, top"""
    rows = []
    for c in df.columns:
        s = df[c]
        top_val = s.dropna().mode().iloc[0] if s.dropna().shape[0] else ''
        rows.append({
            'col': c,
            'dtype': str(s.dtype),
            'missing': int(s.isnull().sum()),
            'unique': int(s.nunique(dropna=True)),
            'top': str(top_val)
        })
    return pd.DataFrame(rows)

Filters = Dict[str, Tuple[str, Any]]

def numeric_cols(df): return df.select_dtypes(include=[np.number]).columns.tolist()
def non_numeric_cols(df): return [c for c in df.columns if c not in numeric_cols(df)]

def apply_filters(df: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    """Apply filters dictionary to a DataFrame"""
    out = df
    for col, (typ, val) in filters.items():
        if col not in out.columns: continue
        if typ == 'range':
            lo, hi = val
            out = out[(pd.to_numeric(out[col], errors='coerce') >= lo) & (pd.to_numeric(out[col], errors='coerce') <= hi)]
        elif typ == 'in':
            out = out[out[col].astype(str).isin(val)]
        elif typ == 'contains':
            out = out[out[col].astype(str).str.contains(val, case=False, na=False)]
    return out

# =========================
# Session state initialization
# =========================
for key in ['orig', 'working', 'filters']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'filters' else {}

# =========================
# Sidebar: data input & filters
# =========================
with st.sidebar:
    st.header('📁 Data Input')
    uploaded = st.file_uploader('CSV or TXT', type=['csv','txt'])
    pasted = st.text_area('Or paste CSV', height=140)

    if st.button('Load', key='load_data'):
        try:
            if uploaded is not None: df_load = read_csv_file(uploaded)
            elif pasted.strip(): df_load = read_csv_file(StringIO(pasted))
            else: st.warning('Provide file or pasted content'); df_load = None
            if df_load is not None: st.session_state.orig = df_load.copy(); st.session_state.working = df_load.copy(); st.success('Data loaded successfully')
        except Exception as e:
            st.error(f'Error loading CSV: {e}')

    if st.session_state.orig is not None and st.button('Reset', key='reset_data'):
        st.session_state.working = st.session_state.orig.copy(); st.session_state.filters = {}

    st.markdown('---')
    st.subheader('🔍 Filters')
    if st.session_state.orig is not None:
        df0 = st.session_state.orig
        if st.checkbox('Show profile', key='show_profile'): st.dataframe(profile(df0), height=250)

        new_filters: Filters = {}
        with st.expander('Column filters'):
            for c in df0.columns:
                s = df0[c]
                st.write(f'**{c}** — {str(s.dtype)}')
                if pd.api.types.is_numeric_dtype(s):
                    mn, mx = float(pd.to_numeric(s).min()), float(pd.to_numeric(s).max())
                    if np.isfinite(mn) and np.isfinite(mx) and mn != mx:
                        rng = st.slider(c, mn, mx, (mn, mx), key=f'flt_{c}')
                        if rng != (mn, mx): new_filters[c] = ('range', rng)
                else:
                    vals = s.dropna().unique()
                    if len(vals) <= 25:
                        sel = st.multiselect(f'{c} values', sorted(map(str, vals)), key=f'multi_{c}')
                        if sel: new_filters[c] = ('in', sel)
                    else:
                        txt = st.text_input(f'Contains ({c})', key=f'txt_{c}')
                        if txt: new_filters[c] = ('contains', txt)

        if st.button('Apply filters', key='apply_filters'): 
            st.session_state.working = apply_filters(st.session_state.orig, new_filters); st.session_state.filters = new_filters; st.success(f'Applied {len(new_filters)} filters')
        if st.button('Clear filters', key='clear_filters'): 
            st.session_state.filters = {}; st.session_state.working = st.session_state.orig.copy(); st.success('Filters cleared')

# =========================
# Stop if no data loaded
# =========================
if st.session_state.working is None:
    st.title('📊 CSV Analyzer'); st.info('Upload or paste a CSV in the sidebar to begin'); st.stop()

# =========================
# Main data reference
# =========================
df = st.session_state.working.copy()
nums = numeric_cols(df)
cats = non_numeric_cols(df)

# =========================
# Tabs: Overview, Cleaning, Visuals, Analytics
# =========================
T1,T2,T3,T4 = st.tabs(['Overview','Cleaning','Visuals','Analytics'])

# =========================
# Tab 1: Overview
# =========================
with T1:
    st.header('Overview')
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Rows', f"{len(df):,}"); c2.metric('Columns', f"{len(df.columns):,}")
    c3.metric('Missing', f"{df.isnull().sum().sum():,}"); c4.metric('Duplicates', f"{df.duplicated().sum():,}")
    st.subheader('Preview'); st.dataframe(df.head(10), use_container_width=True)
    st.subheader('Profile'); st.dataframe(profile(df), use_container_width=True)

# =========================
# Tab 2: Cleaning
# =========================
with T2:
    st.header('Cleaning')
    strategy = st.selectbox('Missing value strategy',['Keep','Drop rows','Fill mean','Fill median','Fill 0'], key='clean_strategy')
    if st.button('Apply missing strategy', key='apply_missing'):
        nums_local = numeric_cols(df)
        if strategy == 'Drop rows': df = df.dropna()
        elif strategy == 'Fill mean': df[nums_local] = df[nums_local].fillna(df[nums_local].mean())
        elif strategy == 'Fill median': df[nums_local] = df[nums_local].fillna(df[nums_local].median())
        elif strategy == 'Fill 0': df[nums_local] = df[nums_local].fillna(0)
        st.session_state.working = df; st.success('Applied missing strategy')
    if st.button('Remove duplicates', key='remove_dupes'): 
        before=len(df); df=df.drop_duplicates(); st.session_state.working=df; st.success(f'Removed {before-len(df)} duplicates')
    drops=st.multiselect('Drop columns', df.columns.tolist(), key='drop_cols')
    if st.button('Drop selected columns', key='drop_btn') and drops: 
        df=df.drop(columns=drops); st.session_state.working=df; st.success(f'Dropped {len(drops)} columns')
    st.markdown('---'); st.dataframe(df.head(), use_container_width=True)
    fmt=st.selectbox('Download format',['csv','excel'], key='download_fmt_clean')
    st.download_button('Download', data=df_to_bytes(df,fmt), file_name=f'cleaned.{"xlsx" if fmt=="excel" else "csv"}', key='download_btn_clean')

# -------------------------
# Visuals Tab
# -------------------------
with T3:
    st.header('Visuals & Exploration')

    kind = st.selectbox('Chart type', 
                        ['Histogram','Scatter','Box','Line','Area',
                         'Bar','Stacked Bar','Pie','Missingness'],
                        key='visual_kind_flex')

    fig = None  # Figure placeholder

    def sel(label, options, default=None):
        if not options: return None
        default = default if default in options else options[0]
        return st.selectbox(label, options, index=options.index(default), key=f'{kind}_{label}')

    def multisel(label, options, default=None):
        default = default if default else options[:min(5,len(options))]
        return st.multiselect(label, options, default=default, key=f'{kind}_{label}_ms')

    all_cols = df.columns.tolist()

    # -------------------------
    # HISTOGRAM
    # -------------------------
    if kind == 'Histogram':
        col = sel('Column', all_cols)
        bins = st.slider('Bins', 5, 100, 20, key=f'{kind}_bins')
        color_col = sel('Color by (optional)', [None]+all_cols)
        fig = px.histogram(df, x=col, color=color_col, nbins=bins, marginal="box", title=f'Histogram of {col}')

    # -------------------------
    # SCATTER
    # -------------------------
    elif kind == 'Scatter':
        x_col = sel('X-axis', all_cols)
        y_col = sel('Y-axis', all_cols)
        color_col = sel('Color by', [None]+all_cols)
        size_col = sel('Size by', [None]+all_cols)
        trendline = st.checkbox('Add trendline', key=f'{kind}_trendline')
        fig = px.scatter(df, x=x_col, y=y_col,
                         color=color_col if color_col else None,
                         size=size_col if size_col else None,
                         trendline='ols' if trendline else None,
                         title=f'{y_col} vs {x_col}')

    # -------------------------
    # BOX / VIOLIN
    # -------------------------
    elif kind == 'Box':
        y_col = sel('Column', all_cols)
        x_col = sel('Group by (optional)', [None]+all_cols)
        fig = px.box(df, y=y_col, x=x_col, points='all', title=f'Boxplot of {y_col}' + (f' by {x_col}' if x_col else ''))

    # -------------------------
    # LINE / AREA
    # -------------------------
    elif kind in ['Line','Area']:
        x_col = sel('X-axis', [None]+all_cols)
        y_col = sel('Y-axis', all_cols)
        color_col = sel('Color by (optional)', [None]+all_cols)
        if kind == 'Line':
            fig = px.line(df, x=x_col if x_col else df.index, y=y_col,
                          color=color_col, title=f'Line Chart of {y_col}' + (f' by {color_col}' if color_col else ''))
        else:
            fig = px.area(df, x=x_col if x_col else df.index, y=y_col,
                          color=color_col, title=f'Area Chart of {y_col}' + (f' by {color_col}' if color_col else ''))

    # -------------------------
    # BAR / STACKED BAR
    # -------------------------
    elif kind in ['Bar','Stacked Bar']:
        x_col = sel('X-axis', all_cols)
        y_col = sel('Y-axis', all_cols)
        color_col = sel('Color by (optional)', [None]+all_cols)
        
        if kind == 'Bar':
            fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                        title=f'Bar Chart: {y_col} vs {x_col}' + (f' colored by {color_col}' if color_col else ''))
        else:  # Stacked Bar
            fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                        title=f'Stacked Bar: {y_col} vs {x_col}' + (f' colored by {color_col}' if color_col else ''))

    # -------------------------
    # PIE
    # -------------------------
    elif kind == 'Pie':
        cat_col = sel('Categorical column', all_cols)
        counts = df[cat_col].value_counts().reset_index()
        counts.columns = [cat_col,'count']
        fig = px.pie(counts, names=cat_col, values='count', title=f'Pie chart of {cat_col}')

    # -------------------------
    # MISSINGNESS
    # -------------------------
    elif kind == 'Missingness':
        missing = df.isnull().sum().reset_index()
        missing.columns = ['Column','Missing']
        fig = px.bar(missing, x='Column', y='Missing', title='Missing Values per Column')

    # -------------------------
    # RENDER FIGURE
    # -------------------------
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # DOWNLOAD DATA
    # -------------------------
    fmt_v = st.selectbox('Download current DataFrame', ['csv','excel'])
    st.download_button('Download Data', 
                       data=df_to_bytes(df, fmt_v), 
                       file_name=f'data_visuals.{"xlsx" if fmt_v=="excel" else "csv"}',)


# -------------------------
# Analytics Tab
# -------------------------
with T4:
    st.header('Analytics')
    st.subheader('Descriptive statistics')
    st.dataframe(df.describe(include='all').T, use_container_width=True)

    st.subheader('Correlations')
    if nums:
        corr_method = st.selectbox('Method', ['pearson','spearman','kendall'], key='corr_method')
        corr = df[nums].corr(method=corr_method)    #type:ignore
        st.dataframe(corr, use_container_width=True)
        fig_corr = px.imshow(corr, text_auto=True, title=f'{corr_method.title()} Correlation Heatmap')
        st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader('Value counts')
    cat_col = st.selectbox('Categorical column', [None]+cats, key='vc_cat')
    if cat_col:
        vc = df[cat_col].value_counts().reset_index()
        vc.columns = [cat_col, 'count']
        st.dataframe(vc, use_container_width=True)
        fig_vc = px.bar(vc, x=cat_col, y='count', title=f'Value counts for {cat_col}')
        st.plotly_chart(fig_vc, use_container_width=True)

    st.subheader('Top N rows')
    n_rows = st.number_input('Number of rows', min_value=1, max_value=len(df), value=10, key='top_n')
    st.dataframe(df.head(n_rows), use_container_width=True)

    fmt_a = st.selectbox('Download current DataFrame', ['csv','excel'], key='dl_analytics_fmt')
    st.download_button('Download Data', data=df_to_bytes(df, fmt_a), file_name=f'data_analytics.{"xlsx" if fmt_a=="excel" else "csv"}', key='dl_analytics_btn')
