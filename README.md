# myTATR：TATR + OCR 表格提取

**运行脚本，自动测试项目自带的十张表格图片。无需选择菜单、拖入图片或填写参数。**

使用微软 TATR 预训练结构模型与 RapidOCR（PP-OCRv6 small / ONNX Runtime），输出带文字的单元格。没有修改模型权重或重新训练。

## 当前电脑怎么运行

在 PyCharm 打开 `D:\Projects\Codex\小毕设\myTATR`，进入底部 **Terminal（终端）**，输入：

```text
.\run.bat
```

CMD 和 PowerShell 都适用。脚本自动选择项目 `.venv-tatr`，或相邻 `PubTables-1M复现/.venv-download` 中已有的 Python 环境。

运行后直接处理十张图片，并显示 `[1/10]` 至 `[10/10]` 的结构识别进度。每次结果放进 `outputs/时间/` 新文件夹。打开其中的 `.html` 查看文字表格，打开 `_overlay.png` 查看识别框；`.csv` 可用 Excel 打开，`.json` 保存完整预测。

PyCharm 已配置好 TATR 解释器时，也可以直接右键运行 `run_table_ocr.py`，同样自动处理这十张。

## 十张图片是什么

图片原文件已从本地下载的 **PubTables-1M 测试集**复制到 [`sample_images/`](sample_images)，来自十篇不同论文：

| 编号 | 类型 | 标注行 × 列 |
|---|---|---:|
| 01 | 紧凑双列表 | 3 × 2 |
| 02 | 长双列表 | 17 × 2 |
| 03 | 分组行标题 | 25 × 3 |
| 04 | 跨行合并单元格 | 20 × 3 |
| 05 | 多列数值矩阵 | 16 × 8 |
| 06 | 纵向长表 | 41 × 3 |
| 07 | 宽表与多级表头 | 10 × 12 |
| 08 | 复杂合并表头 | 11 × 8 |
| 09 | 长段落文字表 | 4 × 4 |
| 10 | 扁平多列表 | 3 × 11 |

这里的行列数用于描述样本，不是模型准确率。[查看图片总览](docs/ten-samples-preview.png)；原始文件名、来源文章和 SHA-256 在 [`sample_images/manifest.json`](sample_images/manifest.json)。图片未修改，XML 标注不参与推理。

v6 升级测试见 [PP-OCRv6 接入与测试](docs/PP-OCRv6接入与测试.md)。原 [默认十图测试报告](docs/十图测试报告.md) 保留为 v4 历史结果。

## 模型和代码怎么连接

```text
sample_images 中的十张图片
  ├─ TATR → 行、列、表头、合并区域
  └─ RapidOCR → 文字、坐标、置信度
               ↓
      单元格重建、文字分配、结果导出
```

TATR 权重仍可使用相邻 `PubTables-1M复现/models/table-transformer-structure-recognition/` 中已有的三个文件，OCR 使用独立的 `.venv-ocr-v6` 环境，首次运行自动下载 PP-OCRv6 small 检测/识别权重到 `models/ocr-v6/`，之后复用本地文件。旧 `.venv-ocr` 环境保留，不再是默认入口。仓库不上传大模型或完整数据集。

当前锁定 `rapidocr==3.10.0`、`onnxruntime==1.31.1`。这里 v6 是 OCR 模型系列，3.10.0 是运行库版本。`ocr.json` 和 `summary.json` 记录实际模型文件名、SHA-256、模型系列和 small 规格。

主要文件：`run_table_ocr.py`（入口）、`ocr_worker.py`（OCR 进程）、`tatr_ocr/fusion.py`（单元格与文字融合）、`tatr_ocr/export.py`（导出）。微软原始后处理源码在 `third_party/tatr/postprocess.py`，未修改；详细原理见 [OCR 接入说明](docs/OCR接入说明.md)。

## 新电脑安装

推荐 Windows x64、Python 3.12。在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
& '.\.venv-tatr\Scripts\python.exe' prepare_model.py
.\run.bat
```

安装、首次准备 TATR 模型和首次运行 OCR 需要联网，权重齐全后从本地文件推理。模型固定 revision 和 SHA-256 校验保存在 `prepare_model.py`；两个环境的依赖版本由 `requirements-*-lock.txt` 锁定。

## 可选：测试其他图片

默认运行无需参数。已有环境中也可继续使用命令行参数：

```text
.\run.bat --images "D:\tables\table1.jpg" "D:\tables\table2.png"
```

或者 `--image-dir "D:\tables"` 处理指定目录内的图片（不递归）。输入文件名须互不重复，不能命名为 `summary` 或 `ocr`。自定义 `--out` 时需指定空目录，避免覆盖结果。

## 测试和边界

### 固定 1000 张双版本评估

`run_benchmark.py` 使用同一份随机测试名单分别评估官方评价条件基线和当前 OCR 系统，计算 AccCon、GriTS Top/Con/Loc。默认十图入口保持不变。

运行方式、数据准备、恢复规则及评价条件见 [千图测试运行说明](docs/千图测试运行说明.md)。本地运行产物保存在忽略上传的 `outputs/benchmark-1000-seed42/`；两套完整的逐表 HTML、CSV、JSON 已打包上传，可从 [千图结果下载](docs/benchmark-1000/full-results/README.md) 获取。不能把导出成功率当成表格内容准确率。

2026-11-11 已完成同一批 1000 张图片的两套评估，均无运行失败：基线的 GriTS Top / Con 为 98.20% / 98.17%，AccCon 为 82.10%；当前 OCR 系统分别为 98.07% / 93.73% 和 7.30%。完整分组结果、置信区间和解释见 [千图双版本测试报告](docs/benchmark-1000/README.md)。

```powershell
& '..\PubTables-1M复现\.venv-download\Scripts\python.exe' -m unittest discover -s tests -v
```

新安装环境可换成 `.venv-tatr\Scripts\python.exe`。单元测试无需模型和完整数据集。

当前输入是裁剪好的论文表格图片，尚不包含整页找表格、手机拍照透视矫正。OCR 对百分号、不等号、上下标和英文空格仍可能出错，长文字也可能跨列；文字框分配数量、置信度不等于 OCR 准确率。

首轮四图集成报告保留在 [`examples/pubtables-smoke/REPORT.md`](examples/pubtables-smoke/REPORT.md)，属于历史测试，与当前默认十张不同。

来源：[Microsoft TATR](https://github.com/microsoft/table-transformer)、[TATR 权重](https://huggingface.co/microsoft/table-transformer-structure-recognition)、[RapidOCR](https://github.com/RapidAI/RapidOCR)、[PubTables-1M](https://huggingface.co/datasets/bsmock/pubtables-1m)。[第三方来源与许可](third_party/README.md)，[样本数据许可](sample_images/DATA_LICENSE.txt)。
