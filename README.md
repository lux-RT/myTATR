# myTATR：TATR + OCR 表格提取

将现成的 TATR 表格结构识别模型与 OCR 接起来，从**已裁剪的表格图片**输出带文字的单元格。

这是独立的整合项目，不是对 TATR 网络的改造：不修改权重、不重新训练。TATR 使用本地 Transformers 格式权重；OCR 使用 **RapidOCR-ONNXRuntime 1.4.4 自带的 PP-OCRv4 模型**，并非安装 PaddleOCR Python 包。

```text
同一张表格图片
  ├─ Transformers / TATR → 行、列、表头、合并区域
  └─ RapidOCR / ONNX Runtime → 文字框、文字、置信度
                          ↓
         官方后处理重建单元格 + 本项目适配层分配文字
                          ↓
              JSON / HTML / CSV / 标注 PNG
```

## 在当前电脑上运行

本地布局如下，数据和模型继续使用此前下载的文件，不复制大文件：

```text
小毕设/
├─ PubTables-1M复现/
│  ├─ .venv-download/                  # 已有 TATR CPU 环境；新增 PyMuPDF 1.26.6
│  ├─ models/table-transformer-structure-recognition/
│  └─ datasets/PubTables-1M-structure/...
└─ myTATR/
   ├─ .venv-ocr/                       # 本次创建的独立 OCR 环境
   └─ run_table_ocr.py
```

在 `myTATR` 文件夹打开 PowerShell，跑四张内置测试样本：

```powershell
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& '..\PubTables-1M复现\.venv-download\Scripts\python.exe' run_table_ocr.py --out outputs/my-test
```

`--out` 必须指向空目录或不存在的目录，避免覆盖已有结果。

识别自己的表格图片：

```powershell
& '..\PubTables-1M复现\.venv-download\Scripts\python.exe' run_table_ocr.py `
  --images 'D:\tables\table1.jpg' 'D:\tables\table2.png' `
  --out outputs/custom-tables
```

`--images` 中的图片文件名须互不重复（忽略大小写），不能命名为 `summary` 或 `ocr`，它们用于批次元数据。

## 从新电脑安装

推荐 Windows x64、Python 3.12，先确认 `python --version`。以下命令会新建两个虚拟环境；CPU 版无需安装 CUDA：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
& '.\.venv-tatr\Scripts\python.exe' prepare_model.py
& '.\.venv-tatr\Scripts\python.exe' run_table_ocr.py `
  --images 'D:\tables\table1.jpg' --out outputs/first-run
```

首次安装/下载需要联网。模型准备完成后，TATR 加载设置为 `local_files_only=True`；OCR 权重随锁定版本的包安装，推理不需要联网。

如已有模型或另一个 OCR 环境，显式指定：

```powershell
& '.\.venv-tatr\Scripts\python.exe' run_table_ocr.py `
  --model-dir 'D:\models\table-transformer-structure-recognition' `
  --ocr-python 'D:\envs\ocr\Scripts\python.exe' `
  --images 'D:\tables\table1.jpg' --out outputs/first-run
```

模型默认先寻找本仓库 `models/table-transformer-structure-recognition`，再寻找相邻的 `PubTables-1M复现/models/...`，均不存在时提示缺失文件。模型固定在 revision `f4d4bdc85c3fe4b1fa49658882a5d38bbdd0f343`，`prepare_model.py` 会校验三个文件的 SHA-256。

## 输出文件怎么用

每张输入生成：

| 文件 | 内容 |
|---|---|
| `图片名.json` | 原始预测框、OCR tokens、单元格坐标与文字、跨行跨列索引、未分配文字、歧义文字索引、耗时 |
| `图片名.html` | 可直接在浏览器打开的文字表格，保留 `rowspan` / `colspan` |
| `图片名.csv` | UTF-8 BOM 表格，可由 Excel 打开；CSV 无合并信息，合并内容只写入左上格 |
| `图片名_overlay.png` | 蓝色为单元格，紫色为合并单元格，绿色为 OCR 文字框 |
| `ocr.json` | 本批次独立 OCR 的原始整合结果 |
| `summary.json` | 所有图片的行列数、文字分配数量和分阶段耗时 |

`row_nums` / `column_nums` 从 0 开始。`ocr_confidence` 是 OCR 模型分数，不能当成准确率；`ambiguous_token_indices` 标记同时覆盖多个单元格的 OCR 行框。

## 添加了哪些代码

| 文件 | 作用 |
|---|---|
| [`run_table_ocr.py`](run_table_ocr.py) | 完整运行入口：检查输入、调用独立 OCR 进程、本地加载 TATR、调用融合和导出 |
| [`ocr_worker.py`](ocr_worker.py) | 初始化 RapidOCR，在 CPU 上检测并识别文字；默认先放大 2 倍；向主程序输出 JSON |
| [`tatr_ocr/ocr_adapter.py`](tatr_ocr/ocr_adapter.py) | 把四边形文字框转成 `bbox` 和 TATR token 字段；按实际横纵缩放比例恢复原图坐标 |
| [`tatr_ocr/fusion.py`](tatr_ocr/fusion.py) | 适配 TATR 类别与官方后处理；先保留结构再填字；按文字面积覆盖比例进行唯一分配；记录表外和歧义文字 |
| [`tatr_ocr/export.py`](tatr_ocr/export.py) | 导出合并单元格 HTML、正确处理换行和引号的 CSV；保留单元格内多行文本 |
| [`tatr_ocr/inputs.py`](tatr_ocr/inputs.py) | 解析本地模型位置，阻止输入文件名与结果/元数据发生冲突 |
| [`prepare_model.py`](prepare_model.py) | 为新环境下载并校验固定版本的 TATR 模型 |
| [`scripts/setup.ps1`](scripts/setup.ps1) | 创建独立 TATR/OCR 环境，按锁定依赖安装 |
| [`third_party/tatr/postprocess.py`](third_party/tatr/postprocess.py) | 直接复用、未修改的微软后处理源码；固定提交、校验值、许可证随仓库保留 |
| [`tests/`](tests) | 坐标、文字归属、空白行、合并表头、HTML/CSV 和文件名冲突回归测试 |

原有 `download_pubtables.py`、数据集、TATR `config.json` / `model.safetensors` 均未修改。新仓库中的业务代码都是新增文件；未修改 TATR 的网络、损失函数或训练逻辑。

## 测试与实际效果

```powershell
& '..\PubTables-1M复现\.venv-download\Scripts\python.exe' -m unittest discover -s tests -v
```

从新电脑安装后，用 `.venv-tatr\Scripts\python.exe` 执行相同命令即可。单元测试不需要模型和数据集。

本地实际测试了四张 PubTables-1M 测试集图片。详见 [实测报告](examples/pubtables-smoke/REPORT.md) 和 [代码接入说明](docs/OCR接入说明.md)。输入图片的 XML 标注不参与推理。

仓库附有这四张小样本及其结果。新电脑无需下载整套数据，也可以复跑：

```powershell
$images = (Get-ChildItem -LiteralPath 'examples/pubtables-smoke/input' -Filter '*.jpg').FullName
& '.\.venv-tatr\Scripts\python.exe' run_table_ocr.py --images $images --out outputs/reproduce
```

已知边界：目前验证的是裁剪好的英文论文表格，不包含整页表格检测或真实拍照场景的透视矫正。长英文词的空格、`≥`、百分号、上下标可能识别错误；OCR 行框也可能跨列。没有完整文字真值，未计算 OCR 准确率、TEDS 或 GriTS，文字框分配数量不代表准确率。

## 来源

- [Microsoft Table Transformer](https://github.com/microsoft/table-transformer)
- [TATR Transformers checkpoint](https://huggingface.co/microsoft/table-transformer-structure-recognition)
- [RapidOCR](https://github.com/RapidAI/RapidOCR)
- [PubTables-1M dataset](https://huggingface.co/datasets/bsmock/pubtables-1m)
- [第三方来源和许可证说明](third_party/README.md)
