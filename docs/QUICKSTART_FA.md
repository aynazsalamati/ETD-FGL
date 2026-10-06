# راهنمای سریع اجرای ETD-FGL

## ۱. ساخت محیط

در پوشه اصلی پروژه PowerShell را باز کن:

```powershell
py -3.12 -m venv .venv
```

نسخه مناسب PyTorch را متناسب با CUDA سیستم نصب کن. سپس:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

یا فایل زیر را اجرا کن:

```text
install_requirements.bat
```

## ۲. اجرای واقعی هر سه دیتاست

```text
run_all_real_datasets.bat
```

این فایل به‌ترتیب Cora، PubMed و Reddit را اجرا می‌کند و در پایان جدول‌ها و شکل‌های چندپنلی را می‌سازد.

اجرای جداگانه:

```text
run_pubmed_real.bat
run_reddit_real.bat
```

## ۳. محل خروجی‌ها

```text
outputs\final_multidataset\federated_cora
outputs\final_multidataset\federated_pubmed
outputs\final_multidataset\federated_reddit
outputs\final_multidataset\cross_dataset_figures
```

برای هر دیتاست، پوشه `comparison` شامل CSV، JSON و شکل چندپنلی با فرمت‌های PNG، PDF و SVG است.

## ۴. اجرای پایپ‌لاین اصلی Cora

```text
run_all_baselines.bat
```

خروجی این بخش در مسیر زیر قرار می‌گیرد:

```text
outputs\federated_cora
```

این پایپ‌لاین قدیمی‌تر و کامل‌ترِ تحلیل Cora است و با پایپ‌لاین هماهنگ سه‌دیتاستی تنظیمات یکسانی ندارد.

## نکته GitHub

پوشه‌های زیر روی سیستم باقی می‌مانند اما به GitHub ارسال نمی‌شوند:

```text
.venv
data
outputs
```

فایل `.gitignore` این موارد، کش‌های پایتون و checkpointها را خودکار کنار می‌گذارد.
