# Data provenance

Source: [Gas Turbine CO and NOx Emission Data Set, UCI dataset 551](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set).

Citation: *Gas Turbine CO and NOx Emission Data Set* [Dataset]. (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C5WC95.

License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). The five CSV files in `raw/` are unchanged copies from UCI. `raw/manifest.json` records their SHA-256 checksums and download URL. `python download_data.py` retrieves them directly, without another repository or tutorial.

The files contain hourly aggregates from one gas turbine in Turkey during 2011–2015. All 36,733 observations are used. Within each year, UCI says rows follow chronological order, but individual timestamps are unavailable.

| Role | Years | Observations |
| --- | --- | ---: |
| Train | 2011–2012 | 15,039 |
| Validation | 2013 | 7,152 |
| Test | 2014–2015 | 14,542 |

This reserves the last two years for testing as UCI requests. The preceding year is a separate validation set, rather than cross-validation. Results are therefore not a direct reproduction of a paper's cross-validation scores.

The original files have 11 columns. This project uses nine operating/ambient measurements as inputs and NOX as its sole target. It excludes CO, the other emission measurement, and excludes the file's year and row identifier. The UCI website's feature count includes additional fields; the explicit nine-column schema in `turbine/config.py` defines this model's inputs.

CDP units are ambiguous: the UCI summary labels compressor discharge pressure as mbar despite values around 10–15. The app labels this as source units and preserves the original scale. No physical-unit conversion is assumed. No missing values are imputed. Loading checks all used values for finiteness.

Exact duplicate rows across splits are checked in the tests. This cannot establish independence: adjacent observations can remain correlated, and the source does not provide equipment group identifiers or timestamps.
