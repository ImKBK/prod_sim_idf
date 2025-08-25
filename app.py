import re
import time
import csv
from collections import Counter
from datetime import datetime
from io import BytesIO

import altair as alt
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.util import ngrams
from rapidfuzz import fuzz

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

st.set_page_config(page_title="Product Similarity & CSV Fixer", layout="centered")

tab1, tab2 = st.tabs(["Product Similarity Analysis", "Fix Misaligned CSV"])

with tab1:
    st.header("Product Similarity Analysis")
    st.info("Upload file with **2 COLUMNS ONLY**: NANKEY and Product Description.")
    uploaded_file = st.file_uploader("Upload Excel file (.xlsx only)", type=["xlsx"], key="similarity_uploader")

    def guess_columns(df):
        possible_nankey = None
        possible_desc = None
        for col in df.columns:
            series = df[col].astype(str)
            numeric_ratio = pd.to_numeric(series, errors='coerce').notna().mean()
            avg_length = series.astype(str).str.len().mean()
            if numeric_ratio > 0.7 and possible_nankey is None:
                possible_nankey = col
            elif avg_length > 5 and possible_desc is None:
                possible_desc = col
        return possible_nankey, possible_desc

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, header=0)
            df.columns = df.columns.astype(str).str.strip().str.upper()

            if df.shape[1] > 2:
                st.warning("File has more than 2 columns. Please upload only NANKEY and PROD_DESC columns.")
                st.stop()

            nankey_col, desc_col = guess_columns(df)

            if not nankey_col or not desc_col:
                st.warning("Unable to detect NANKEY and PROD_DESC automatically.")
                nankey_col = st.selectbox("Select NANKEY column", df.columns)
                desc_col = st.selectbox("Select PROD_DESC column", df.columns)
            else:
                st.info(f"Auto-detected columns: NANKEY → **{nankey_col}**, PROD_DESC → **{desc_col}**")

            df = df[[nankey_col, desc_col]]
            df.columns = ["NANKEY", "PROD_DESC"]
            df = df.dropna(subset=["PROD_DESC"]).reset_index(drop=True)

            threshold = st.slider("Select Similarity Threshold (0-100)", min_value=0, max_value=100, value=60, step=1)

            if st.button("Identify Similar Items"):
                df["PROD_DESC"] = df["PROD_DESC"].astype(str).str.lower().str.strip()
                descriptions = df["PROD_DESC"].tolist()
                n = len(descriptions)

                used_indices = set()
                bulk_groups = []
                unique_items = []

                start_time = time.time()
                progress = st.progress(0)

                for i in range(n):
                    if i in used_indices:
                        continue
                    group = [i]
                    for j in range(i + 1, n):
                        if j in used_indices:
                            continue
                        score = fuzz.token_sort_ratio(descriptions[i], descriptions[j])
                        if score >= threshold:
                            group.append(j)
                            used_indices.add(j)
                    if len(group) > 1:
                        bulk_groups.append(group)
                        used_indices.update(group)
                    else:
                        unique_items.append(i)
                    progress.progress((i + 1) / n)

                bulk_data = []
                bulk_group_id = 1
                for group in bulk_groups:
                    for idx in group:
                        nankey = df.loc[idx, 'NANKEY']
                        description = df.loc[idx, 'PROD_DESC']
                        bulk_data.append([bulk_group_id, nankey, description])
                    bulk_group_id += 1

                unique_data = []
                for idx in unique_items:
                    nankey = df.loc[idx, 'NANKEY']
                    description = df.loc[idx, 'PROD_DESC']
                    unique_data.append([nankey, description])

                bulk_df = pd.DataFrame(bulk_data, columns=['Bulk Group ID', 'NANKEY', 'PROD_DESC'])
                unique_df = pd.DataFrame(unique_data, columns=['NANKEY', 'PROD_DESC'])

                pivot_df = bulk_df.groupby('Bulk Group ID').size().reset_index(name='Count of Bulk Group ID')
                pivot_df = pivot_df.sort_values(by='Count of Bulk Group ID', ascending=False)

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
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                end_time = time.time()

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Unique Items", len(unique_items))
                col2.metric("Bulk Groups", len(bulk_groups))
                col3.metric("Bulk Items", sum(len(group) for group in bulk_groups))
                col4.metric("Execution Time (s)", f"{end_time - start_time:.2f}")

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
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"Error fixing CSV: {e}")
