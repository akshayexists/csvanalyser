# 📊 CSV Analyzer

A streamlit app written as a way to circumvent writing fresh code on csv files each time I encounter them.

## Features

- **Data Input**: Upload a CSV/TXT file or paste CSV content directly.  
- **Filtering**: Apply numeric ranges, categorical selections, or text search.  
- **Cleaning**: Handle missing values, remove duplicates, and drop columns.  
- **Visuals**: Interactive charts including Histogram, Scatter (with color/size/trendline), Box, Violin, Line, Area, Bar, Stacked Bar, Pie, Heatmap, Missingness, and Scatter Matrix.  
- **Analytics**: Descriptive stats, correlations, value counts, top N rows.  
- **Export**: Download cleaned/filtered datasets in CSV or Excel.

## Structure

- **Sidebar** – Load data; reset and manage filters.  
- **Tabs**
  1. **Overview** – Dataset summary, profile, and preview.  
  2. **Cleaning** – Missing values, duplicates, drop columns.  
  3. **Visuals** – All interactive charts for exploration.  
  4. **Analytics** – Descriptive statistics and correlations.  

## Usage

1. Launch with:  
   ```bash
   streamlit run main.py
   ```

2. Upload or paste your CSV data.

---
