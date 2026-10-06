# راهنمای آزمایش‌های تکمیلی ETD-FGL

این نسخه پنج آزمایش تکمیلی را به پروژه اضافه می‌کند و هیچ عدد یا نمودار فرضی تولید نمی‌کند. نمودار نهایی فقط از `final_summary.json`های واقعی ساخته می‌شود.

## فایل‌های اجرا

1. `run_extended_multiseed.bat`
   - Seedهای 42، 43 و 44
   - همه روش‌های اصلی
   - خروجی mean ± std

2. `run_extended_noniid.bat`
   - Dirichlet α = 0.1, 0.5, 1.0
   - مقایسه Attacked FedAvg، PD-FL، FLPurifier-GNN و ETD-FGL

3. `run_extended_ablation.bat`
   - Full ETD-FGL
   - بدون Topology
   - بدون Explanation
   - بدون Update Anomaly
   - بدون Trust Weighting

4. `run_extended_sensitivity.bat`
   - Poison rate = 0.05, 0.10, 0.20, 0.30
   - Trigger size = 2, 3, 4, 5

5. `run_extended_clients.bat`
   - 5، 10، 20، 30 و 50 کلاینت

برای اجرای همه موارد پشت سر هم:

`run_all_extended_experiments.bat`

این اجرای کامل طولانی است. روی GTX 1050 بهتر است فایل‌ها را یکی‌یکی اجرا کنی تا در صورت خطا، دقیقاً همان مطالعه را اصلاح کنیم.

## مسیر خروجی

خروجی خام:

`outputs\extended_experiments`

جدول‌ها و شکل‌های نهایی ژورنالی:

`outputs\extended_experiments\journal_summary`

## شکل‌های نهایی

- `Fig_Ext_01_MultiSeed_MeanStd`
- `Fig_Ext_02_NonIID_Dirichlet`
- `Fig_Ext_03_ETDFGL_Ablation`
- `Fig_Ext_04_Sensitivity_PoisonRate`
- `Fig_Ext_05_Sensitivity_TriggerSize`
- `Fig_Ext_06_Client_Scalability`

هر شکل در PNG، PDF و SVG ذخیره می‌شود.
