import re
import time
import csv
from collections import Counter, defaultdict, deque
from datetime import datetime
from io import BytesIO

import altair as alt
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.util import ngrams
from rapidfuzz import fuzz, process as rf_process

# Optional fast path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# =========================
# Setup
# =========================
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

st.set_page_config(page_title="Product Similarity & CSV Fixer", layout="wide")

# =========================
# Helper Functions
# =========================

def guess_columns(df: pd.DataFrame):
    possible_nankey = None
    possible_desc = None
    for col in df.columns:
        series = df[col].astype(str)
        numeric_ratio = pd.to_numeric(series, errors='coerce').notna().mean()
        avg_length = series.astype(str).str.len().mean()
        # Heuristic: ID-like column is mostly numeric/short, description is longer text
        if numeric_ratio > 0.7 and possible_nankey is None:
            possible_nankey = col
        elif avg_length > 5 and possible_desc is None:
            possible_desc = col
    # Fallbacks if heuristics fail
    if possible_nankey is None and len(df.columns) >= 1:
        possible_nankey = df.columns[0]
    if possible_desc is None and len(df.columns) >= 2:
        possible_desc = df.columns[1]
    return possible_nankey, possible_desc


def normalize_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def build_groups_from_edges(n: int, edges: list[tuple[int, int]]):
    """Union-Find via BFS to build connected components from pair edges."""
    adj = defaultdict(list)
    for a, b in edges:
        if a == b:
            continue
        adj[a].append(b)
        adj[b].append(a)
    visited = [False] * n
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        queue = deque([i])
        visited[i] = True
        comp = [i]
        while queue:
            u = queue.popleft()
            for v in adj.get(u, []):
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
                    comp.append(v)
        groups.append(sorted(comp))
    return groups


def fuzzy_grouping(descriptions: list[str], threshold: int, progress_cb=None):
    """Memory-safe fuzzy grouping using RapidFuzz with candidate blocking.
    Blocking: index by first 2 non-stopword tokens to limit comparisons.
    """
    # Build blocks
    token_blocks = defaultdict(list)
    for idx, text in enumerate(descriptions):
        tokens = re.findall(r"[a-z0-9]+", text)
        tokens = [t for t in tokens if t not in stop_words]
        key = tuple(tokens[:2]) if len(tokens) >= 2 else tuple(tokens)
        token_blocks[key].append(idx)

    n = len(descriptions)
    edges = []
    processed = 0

    for key, idxs in token_blocks.items():
        # Within each block do pairwise via extract to avoid O(m^2) loops
        if len(idxs) == 1:
            processed += 1
            if progress_cb:
                progress_cb(processed / n)
            continue
        # Build a local list for RF lookup
        local_descs = [descriptions[i] for i in idxs]
        for local_i, desc in enumerate(local_descs):
            matches = rf_process.extract(
                desc,
                local_descs,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=threshold,
                limit=None,
            )
            i_global = idxs[local_i]
            for _, score, j_local in matches:
                if j_local == local_i:
                    continue
                j_global = idxs[j_local]
                if i_global < j_global:
                    edges.append((i_global, j_global))
            processed += 1
            if progress_cb:
                progress_cb(processed / n)

    groups = build_groups_from_edges(n, edges)
    return groups


def tfidf_grouping(descriptions: list[str], cosine_threshold: float, progress_cb=None):
    """Fast, scalable text grouping using TF-IDF + cosine radius neighbors.
    Builds edges for pairs above cosine_threshold and returns connected components.
    """
    # Vectorize
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )
    X = vectorizer.fit_transform(descriptions)

    n = X.shape[0]
    # Cosine distance = 1 - cosine_similarity
    radius = 1.0 - float(cosine_threshold)

    # NearestNeighbors on sparse TF-IDF
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(X)

    # Query in batches to control memory
    batch = 2000
    edges = []
    for start in range(0, n, batch):
        end = min(start + batch, n)
        dists, inds = nn.radius_neighbors(X[start:end], radius=radius, return_distance=True)
        for offset, (dist_list, ind_list) in enumerate(zip(dists, inds)):
            i = start + offset
            for d, j in zip(dist_list, ind_list):
                if i == j:
                    continue
                # Convert dist back to cosine similarity and check
                sim = 1.0 - float(d)
                if sim >= cosine_threshold:
                    if i < j:
                        edges.append((i, j))
        if progress_cb:
            progress_cb(end / n)

    groups = build_groups_from_edges(n, edges)
    return groups


def build_outputs(df: pd.DataFrame, groups: list[list[int]]):
    # Separate singletons vs bulk groups
    bulk_groups = [g for g in groups if len(g) > 1]
    singletons = [g[0] for g in groups if len(g) == 1]

    bulk_data = []
    for gid, group in enumerate(bulk_groups, start=1):
        for idx in group:
            bulk_data.append([gid, df.loc[idx, 'NANKEY'], df.loc[idx, 'PROD_DESC']])

    unique_data = [[df.loc[idx, 'NANKEY'], df.loc[idx, 'PROD_DESC']] for idx in singletons]

    bulk_df = pd.DataFrame(bulk_data, columns=['Bulk Group ID', 'NANKEY', 'PROD_DESC'])
    unique_df = pd.DataFrame(unique_data, columns=['NANKEY', 'PROD_DESC'])

    if not bulk_df.empty:
        pivot_df = (
            bulk_df.groupby('Bulk Group ID').size().reset_index(name='Count of Bulk Group ID')
            .sort_values(by='Count of Bulk Group ID', ascending=False)
        )
    else:
        pivot_df = pd.DataFrame(columns=['Bulk Group ID', 'Count of Bulk Group ID'])

    return bulk_df, unique_df, pivot_df


def bigram_chart_and_download(unique_df: pd.DataFrame):
    st.subheader("Top 10 Frequent Word Pairs in Unique Items")
    if unique_df.empty:
        st.info("No Unique Items available for analysis.")
        return

    desc_series = unique_df['PROD_DESC'].dropna().astype(str)
    all_bigrams = []
    for desc in desc_series:
        tokens = re.findall(r'\b[a-z]{2,}\b', desc.lower())
        filtered = [w for w in tokens if w not in stop_words]
        bigrams = list(ngrams(filtered, 2))
        all_bigrams.extend(bigrams)

    bigram_counts = Counter(all_bigrams)
    if not bigram_counts:
        st.info("No frequent bigrams found in Unique Items.")
        return

    excluded_bigrams = {('fl', 'oz')}
    filtered_bigrams = [(pair, count) for pair, count in bigram_counts.items() if pair not in excluded_bigrams]
    all_bigrams_with_counts = sorted(filtered_bigrams, key=lambda x: x[1], reverse=True)

    top10 = all_bigrams_with_counts[:10]
    top10_df = pd.DataFrame([(f"{a} {b}", c) for (a, b), c in top10], columns=["Bigram", "Count"])

    chart = alt.Chart(top10_df).mark_bar().encode(
        x=alt.X('Count:Q'),
        y=alt.Y('Bigram:N', sort='-x'),
        tooltip=['Bigram', 'Count']
    ).properties(
        title='Top 10 Bigrams (Unique Items)',
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

    all_bigrams_df = pd.DataFrame(
        [(f"{a} {b}", c) for (a, b), c in all_bigrams_with_counts],
        columns=["Bigram", "Count"]
    )
    csv_output = all_bigrams_df.to_csv(index=False).encode('utf-8')
    today_str = datetime.today().strftime('%d%m%y')
    st.download_button(
        " Download Word Pairs",
        data=csv_output,
        file_name=f"bigrams_unique_items_{today_str}.csv",
        mime='text/csv'
    )


# =========================
# UI: Tabs
# =========================
tab1, tab2 = st.tabs(["Product Similarity Analysis", "Fix Misaligned CSV"]) 

with tab1:
    st.header("Product Similarity Analysis")
    st.info("Upload file with **2 COLUMNS ONLY**: NANKEY and Product Description.")

    uploaded_file = st.file_uploader("Upload Excel file (.xlsx only)", type=["xlsx"], key="similarity_uploader")

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, header=0)
            df.columns = df.columns.astype(str).str.strip()

            # Enforce exactly 2 columns
            if df.shape[1] != 2:
                st.warning("File must have exactly 2 columns: NANKEY and Product Description.")
                st.stop()

            nankey_col, desc_col = guess_columns(df)
            st.info(f"Auto-detected columns: NANKEY → **{nankey_col}**, PROD_DESC → **{desc_col}**")

            df = df[[nankey_col, desc_col]].copy()
            df.columns = ["NANKEY", "PROD_DESC"]
            df = df.dropna(subset=["PROD_DESC"]).reset_index(drop=True)

            # Normalize once
            df["PROD_DESC"] = normalize_text(df["PROD_DESC"])
            descriptions = df["PROD_DESC"].tolist()

            # Controls
            method = st.radio(
                "Method",
                ["TF-IDF (fast)", "Fuzzy (token_sort_ratio)"],
                index=0,
                help="TF-IDF is recommended for large files (10k+). Fuzzy is more flexible but slower.",
            )

            if method == "TF-IDF (fast)":
                cosine_thr = st.slider(
                    "Cosine similarity threshold",
                    min_value=0.50,
                    max_value=0.95,
                    value=0.80,
                    step=0.01,
                    help="Higher = stricter grouping. 0.80 is a reasonable start.",
                )
            else:
                fuzzy_thr = st.slider(
                    "Fuzzy similarity threshold (0-100)",
                    min_value=0,
                    max_value=100,
                    value=60,
                    step=1,
                )

            run = st.button("Identify Similar Items")

            if run:
                start_time = time.time()
                progress = st.progress(0)

                if method == "TF-IDF (fast)":
                    groups = tfidf_grouping(descriptions, cosine_thr, progress_cb=progress.progress)
                else:
                    groups = fuzzy_grouping(descriptions, fuzzy_thr, progress_cb=progress.progress)

                bulk_df, unique_df, pivot_df = build_outputs(df, groups)

                # Export
                output = BytesIO()
                today_str = datetime.today().strftime('%d%m%y')
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    unique_df.to_excel(writer, sheet_name='Unique Items', index=False)
                    bulk_df.to_excel(writer, sheet_name='Bulk Items', index=False)
                    pivot_df.to_excel(writer, sheet_name='Pivots', index=False)

                st.success("Download your output below.")
                st.download_button(
                    label="Download Excel Output",
                    data=output.getvalue(),
                    file_name=f"bulk_identifier_output_{today_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                end_time = time.time()

                col1, col2, col3, col4 = st.columns(4)
                unique_count = sum(1 for g in groups if len(g) == 1)
                bulk_groups_count = sum(1 for g in groups if len(g) > 1)
                bulk_items_count = sum(len(g) for g in groups if len(g) > 1)

                col1.metric("Unique Items", unique_count)
                col2.metric("Bulk Groups", bulk_groups_count)
                col3.metric("Bulk Items", bulk_items_count)
                col4.metric("Execution Time (s)", f"{end_time - start_time:.2f}")

                st.subheader("Charts")

                # Unique vs Bulk pie
                data = pd.DataFrame({
                    'Type': ['Unique Items', 'Bulk Items'],
                    'Count': [unique_count, bulk_items_count]
                })
                pie_chart = alt.Chart(data).mark_arc(innerRadius=50).encode(
                    theta='Count:Q',
                    color='Type:N',
                    tooltip=['Type', 'Count']
                ).properties(title='Unique vs Bulk Items Distribution')
                st.altair_chart(pie_chart, use_container_width=True)

                # Top bulk groups bar
                if not pivot_df.empty:
                    top_bulk = pivot_df.head(10)
                    chart = alt.Chart(top_bulk).mark_bar().encode(
                        x=alt.X('Bulk Group ID:O', sort='-y', title='Bulk Group ID'),
                        y=alt.Y('Count of Bulk Group ID:Q', title='Items in Group'),
                        tooltip=['Bulk Group ID', 'Count of Bulk Group ID']
                    ).properties(title='Top 10 Bulk Groups by Size', width=600, height=400)
                    st.altair_chart(chart, use_container_width=True)

                # Bigrams on Unique Items
                bigram_chart_and_download(unique_df)

        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab2:
    st.header("Fix Misaligned CSV Rows")
    st.info("Upload a misaligned **CSV file**")

    csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="csv_fixer")
    expected_columns = 7

    if csv_file:
        try:
            cleaned_rows = []
            reader = csv.reader(csv_file.read().decode('utf-8').splitlines())
            header = next(reader)
            cleaned_rows.append(header)

            for row in reader:
                if len(row) > expected_columns:
                    fixed_row = [row[0]]
                    merge_count = len(row) - expected_columns + 1
                    description = ','.join(row[1:1 + merge_count])
                    fixed_row.append(description)
                    fixed_row += row[1 + merge_count:]
                    if len(fixed_row) == expected_columns:
                        cleaned_rows.append(fixed_row)
                elif len(row) == expected_columns:
                    cleaned_rows.append(row)
                else:
                    continue

            df_cleaned = pd.DataFrame(cleaned_rows[1:], columns=cleaned_rows[0])

            output_cleaned = BytesIO()
            df_cleaned.to_excel(output_cleaned, index=False)
            st.success("File cleaned successfully.")
            st.download_button(
                label="Download Cleaned Excel",
                data=output_cleaned.getvalue(),
                file_name="cleaned_file.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            st.error(f"Error fixing CSV: {e}")
