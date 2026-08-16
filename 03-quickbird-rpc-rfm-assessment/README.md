# QuickBird-2 RPC/RFM Geometric Accuracy Assessment

ارزیابی مدل رشنال زمین‌وابسته و ضرایب RPC زمین‌مستقل برای داده QuickBird-2 و پالایش بایاس.

## Methods
UTM Zone 10 to geodetic transformation, normalization, terrain-dependent RFM estimation, vendor RPC evaluation, indirect bias refinement and GCP/ICP RMSE comparison.

## Included inputs
- `../../data/quickbird/GCPsData.txt` — 84 observations
- `../../data/quickbird/RPC.RPB` — QuickBird-2 RPC00B metadata

## Deliverables
- `notebooks/quickbird_rpc_rfm_assessment.ipynb`
- `src/quickbird_rpc_rfm_assessment.py`
- `reports/quickbird_rpc_rfm_report_redacted.pdf`
