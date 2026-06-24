# Datasets

This folder documents the datasets used in this research project.
Due to size constraints, raw data files are **not stored in this repository**.

## Datasets Used

### 1. UCDP/PRIO Armed Conflict Dataset v26.1
- **Source:** Uppsala Conflict Data Program (UCDP) / Peace Research Institute Oslo (PRIO)
- **File:** `UcdpPrioConflict_v26_1.csv`
- **Size:** ~507 KB
- **Coverage:** 1946–2024, country-year dyadic armed conflict data
- **Download:** https://ucdp.uu.se/downloads/
- **Citation:** Gleditsch, N. P., Wallensteen, P., Eriksson, M., Sollenberg, M., & Strand, H. (2002). Armed Conflict 1946-2001: A New Dataset. *Journal of Peace Research*, 39(5), 615–637.

### 2. World Bank Development Indicators
- **File:** `4a217978-8c5c-491a-88a5-2cba7f5f914e_Data.csv`
- **Size:** ~2 MB
- **Key Indicators Used:**
  - GDP per capita (NY.GDP.PCAP.CD)
  - GDP growth (NY.GDP.MKTP.KD.ZG)
  - Inflation (FP.CPI.TOTL.ZG)
  - Unemployment (SL.UEM.TOTL.ZS)
  - Population (SP.POP.TOTL)
  - Urban population % (SP.URB.TOTL.IN.ZS)
  - Internet users % (IT.NET.USER.ZS)
  - Trade % of GDP (NE.TRD.GNFS.ZS)
- **Download:** https://databank.worldbank.org/source/world-development-indicators

### 3. V-Dem (Varieties of Democracy) Dataset v16
- **File:** `V-Dem-CY-Full+Others-v16.csv`
- **Size:** ~387.5 MB
- **Key Variables Used:**
  - v2x_polyarchy (Electoral democracy)
  - v2x_libdem (Liberal democracy)
  - v2xcl_rol (Rule of law)
  - v2x_liberal (Liberal component)
  - v2x_partip (Participatory component)
  - v2x_freexp_altinf (Freedom of expression)
  - v2x_frassoc_thick (Freedom of association)
  - v2x_cspart (Civil society participation)
- **Download:** https://www.v-dem.net/data/the-v-dem-dataset/
- **Citation:** Coppedge, M., et al. (2024). V-Dem [Country-Year/Country-Date] Dataset v14. *Varieties of Democracy (V-Dem) Project*.

## Data Access via Google Drive

All raw datasets are available on Google Drive:
https://drive.google.com/drive/folders/1UDvdjrnlStdHAHjz0oVnAF9ZJEFALV24

## Data Preprocessing

See the notebook `notebooks/geopolitical_conflict_analysis.ipynb` for full preprocessing steps including:
- UCDP conflict-year lag feature engineering
- World Bank wide-to-long format conversion
- V-Dem country-year join
- Country name harmonisation across datasets
- Temporal train/test splitting (cutoff: 2010)
