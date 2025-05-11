# AGTransformer_PointCloud
Attention Graph Transformer on PointCloud Data



یک فایل نصب خودکار (requirements.txt) و همچنین یک اسکریپت ساده Bash/Powershell آماده می‌کنم که تنها با یک دستور، تمام پکیج‌ها شامل torch، torchvision، torchaudio، و تمام ماژول‌های مورد نیاز PyTorch Geometric را به‌صورت خودکار نصب کند.


# فایل‌های نصب پکیج‌ها برای PyTorch 2.7.0+CPU

برای نصب PyTorch 2.7.0 (نسخه CPU-only) به همراه بسته‌های مرتبط (torchvision و torchaudio) می‌توان از مخزن رسمی PyTorch استفاده کرد. راهنمای رسمی نشان می‌دهد که برای حالت **بدون GPU** باید از شاخص `cpu` در URL مربوط به PyTorch استفاده کنیم. به عنوان مثال در دستور زیر، نسخه‌های متناسب از PyTorch، torchvision و torchaudio را نصب می‌کنیم:

```bash
pip install torch==2.7.0+cpu torchvision==0.22.0+cpu torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu
```

بسته‌های پایه‌ی پایتون مانند **numpy**، **scikit-learn** و **h5py** را می‌توان مستقیماً از PyPI نصب کرد. بنابراین در فایل `requirements.txt` فقط نام آن‌ها را قید می‌کنیم (نسخه‌ را در صورت نیاز می‌توان ثابت کرد). به طور مثال:

```
numpy
scikit-learn
h5py
```

برای نصب **PyTorch Geometric** (PyG) و کتابخانه‌های کمکی آن (مانند `torch_scatter`, `torch_sparse`, `torch_cluster`, `pyg_lib`, `torch_spline_conv`)، باید از مخزن wheelهای پیش‌ساخته‌ی PyG استفاده کرد. در راهنمای PyG نشان داده شده است که با تنظیم متغیرها `${TORCH}` روی نسخه PyTorch و `${CUDA}` روی `cpu`، می‌توانیم لینک wheel مناسب را به pip بدهیم. برای PyTorch 2.7.0 (CPU) لینک زیر را به کار می‌بریم:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch-geometric \
    -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
```

در اینجا `-f https://data.pyg.org/whl/torch-2.7.0+cpu.html` باعث می‌شود که pip پکیج‌های مورد نیاز PyG را از آن URL دریافت کند (از جمله بسته‌های ذکرشده). بدین ترتیب تمامی نیازمندی‌ها با یک دستور `pip install` قابل دستیابی هستند.

## محتویات فایل `requirements.txt`

محتویات نهایی `requirements.txt` که تمامی بسته‌های ذکرشده را شامل می‌شود، می‌تواند به صورت زیر باشد:

```
# استفاده از PyPI پیش‌فرض برای numpy، scikit-learn، h5py
numpy
scikit-learn
h5py

# افزودن شاخص CPU برای PyTorch و وابستگی‌های آن
--index-url https://download.pytorch.org/whl/cpu
torch==2.7.0+cpu
torchvision==0.22.0+cpu
torchaudio==2.7.0

# افزودن لینک wheelهای PyG برای PyTorch 2.7.0 CPU
--find-links https://data.pyg.org/whl/torch-2.7.0+cpu.html
pyg_lib
torch-scatter
torch-sparse
torch-cluster
torch-spline-conv
torch-geometric
```

برای نصب این پکیج‌ها کافی است دستور زیر را اجرا کنید:

```bash
pip install -r requirements.txt
```

### اسکریپت نصب جایگزین (install.sh)

در صورت تمایل می‌توانید بجای فایل `requirements.txt` از یک اسکریپت شل استفاده کنید. به عنوان مثال، محتویات فایل `install.sh` به شکل زیر خواهد بود:

```bash
#!/bin/bash
# نصب PyTorch، torchvision، torchaudio (نسخه‌های CPU)
pip install torch==2.7.0+cpu torchvision==0.22.0+cpu torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu

# نصب بسته‌های عمومی
pip install numpy scikit-learn h5py

# نصب PyTorch Geometric و وابسته‌ها از wheelهای PyG
pip install pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
```

و روی ویندوز می‌توانید از اسکریپت مشابه PowerShell (`install.ps1`) استفاده کنید که حاوی همین دستورات pip باشد.

**منابع:** دستور نصب CPU-only در مستندات رسمی PyTorch ذکر شده است و راهنمای نصب PyTorch Geometric نحوه استفاده از URL wheelهای PyG را توضیح می‌دهد.

