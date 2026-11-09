# PP-OCRv6 接入与测试（2026-11-09）

当前默认 OCR 已从 PP-OCRv4 升级为 **PP-OCRv6 small 检测模型 + PP-OCRv6 small 识别模型**。采用 RapidOCR 提供的 ONNX 转换权重，在 Windows CPU 上运行。small 是 v6 的模型规格；本次没有训练或微调模型。

## 版本与环境

| 项目 | 升级前 | 当前 |
|---|---|---|
| Python OCR 包 | rapidocr-onnxruntime 1.4.4 | rapidocr 3.10.0 |
| 模型系列 | PP-OCRv4 | PP-OCRv6 small（检测、识别均已切换） |
| 推理引擎 | ONNX Runtime 1.23.2 | ONNX Runtime 1.31.1 |
| 独立虚拟环境 | `.venv-ocr` | `.venv-ocr-v6` |
| 模型位置 | 旧安装包内 | `models/ocr-v6/` |

核对当日 PyPI 发布版本后固定依赖，完整传递依赖在 `requirements-ocr-lock.txt`。旧环境保留供对照，默认入口明确使用新环境，不自动回退到 v4。TATR 仍由原来的 PyTorch 2.8.0+cpu / Transformers 4.57.1 环境运行，通过子进程 JSON 交换 OCR 结果。

官方说明：[PP-OCRv6](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)、[RapidOCR 模型选择](https://rapidai.github.io/RapidOCRDocs/main/model_list/)。RapidOCR 是运行封装，PP-OCRv6 是模型系列，两者版本号不同。

## 实际加载的模型

首次运行自动从 RapidAI 的 ModelScope `v3.10.0` 发布目录下载并校验；再次运行复用本地文件。本机下载已完成，无需重复下载。大权重及虚拟环境不提交 GitHub。

| 文件 | SHA-256 |
|---|---|
| `PP-OCRv6_det_small.onnx` | `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f` |
| `PP-OCRv6_rec_small.onnx` | `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` |

完整版本化下载 URL、实际加载状态及哈希在 [十图 summary](ppocr-v6-ten-summary.json) 的 `ocr_models` 中，也会写入每次输出的 `ocr.json` / `summary.json`。配置明确启用 v6 small 检测、识别，关闭方向分类器，采用 CPU 四线程、外部放大 2 倍。方向分类器没有加载旧版模型。

## 修改了哪些代码

| 文件 | 本次修改 |
|---|---|
| `ocr_worker.py` | 切换新版 `rapidocr` API；显式选择 v6 small 检测/识别及 CPU 参数；缓存模型；记录实际模型路由、加载状态和文件哈希；适配新版分阶段计时。 |
| `tatr_ocr/ocr_adapter.py` | 新增 `normalize_result()`，把新版 `boxes/txts/scores` 转成原有 token；检查结果数组长度和完整性；将 NumPy 坐标转为可序列化的 Python 浮点数，保留原图坐标还原。 |
| `run_table_ocr.py` | 默认 OCR 解释器切到 `.venv-ocr-v6`；汇总结果包含 OCR 模型系列、规格和文件记录。 |
| `requirements-ocr-v6.txt`、`requirements-ocr.txt`、`requirements-ocr-lock.txt` | 锁定新包及完整依赖；默认 OCR 安装指向 v6。 |
| `scripts/setup.ps1` | 新电脑创建并安装 `.venv-ocr-v6`，提示首次 OCR 运行需要下载模型。 |
| `tests/test_ocr_adapter.py`、`tests/test_default_samples.py` | 覆盖新版结果、空结果、数组缺项/长度不符、JSON 输出和默认新解释器。 |
| README、OCR 接入说明及第三方说明 | 更新环境与下载方式，旧测试报告标为历史记录。 |

TATR 权重、预处理、官方后处理源码、`fusion.py`、导出算法及十张原图保持原样。十图 TATR 原始结构预测与旧 v4 测试逐项相同；融合后的文字变化来自新 OCR 输出。

## 十图完整链路实测

在项目根目录运行：

```powershell
.\run.bat --out outputs/ppocr-v6-ten
```

退出码 0。十张图均生成 JSON、HTML、CSV 和框图，本地结果在 `outputs/ppocr-v6-ten/`；可复核的版本、分阶段耗时和计数保存在 [ppocr-v6-ten-summary.json](ppocr-v6-ten-summary.json)。

| 编号 | 预测行 × 列 | v4 OCR 框数 | v6 OCR 框数 | v6 入表框数 | v6 几何歧义框数 |
|---|---:|---:|---:|---:|---:|
| 01 | 3 × 2 | 10 | 10 | 6 | 0 |
| 02 | 17 × 2 | 35 | 35 | 34 | 0 |
| 03 | 25 × 3 | 93 | 94 | 66 | 1 |
| 04 | 20 × 3 | 53 | 53 | 49 | 0 |
| 05 | 16 × 8 | 104 | 104 | 102 | 0 |
| 06 | 41 × 3 | 127 | 127 | 125 | 4 |
| 07 | 10 × 12 | 105 | 112 | 110 | 0 |
| 08 | 11 × 8 | 57 | 58 | 55 | 0 |
| 09 | 4 × 4 | 97 | 96 | 95 | 1 |
| 10 | 3 × 11 | 33 | 33 | 32 | 0 |

v6 合计 722 个文字框，674 个分配入表。表外标题、脚注等仍保存在未分配结果中。行列数与独立标注对照一致；这不代表所有文字和合并关系正确。**框更多、成功分配更多或置信度更高都不能当作文字准确率提升的证据。**

自动检查通过：十张图片源文件哈希、四类导出文件、CSV 网格尺寸及换行、单元格完整且无重复占格、OCR token 唯一分配或未分配可追溯、两个 v6 模型哈希、TATR 三个文件及官方后处理源码哈希。22 项单元测试通过，两个运行环境 `pip check` 通过；另用真实新版结果对象与 NumPy 浮点数组检查了 JSON 序列化。

## 已知错字的额外抽查

另用相同放大倍数、线程数和新 worker 复测历史四图的 OCR 阶段，输出在本地 `outputs/ppocr-v6-historical-four/ocr.json`，未人工改写预测。

| 图片 | 原报告记录的 v4 错误 | 本次 v6 输出 |
|---|---|---|
| `PMC1064082_table_1` | `≥50` 被识别为 `250` | `≥50` |
| `PMC1064082_table_1` | `16(33%)` 被识别为 `16(339)` | `16 (33%)` |
| `PMC1064078_table_0` | `Controls (%)` 被识别为 `Controis(9s)` | `Controls (%)` |

这是已知个例的抽查，未计算整套文字准确率、TEDS 或 GriTS，也没有证明所有符号、跨列归属都改善。当前仍有几何歧义框，后续后处理改进及独立评估仍需完成。模型升级本身属于基础系统更新，不应直接写作已验证的算法创新。

## 现在怎么用

当前电脑已安装环境和下载模型。在 PyCharm 打开 `myTATR` 项目，终端仍输入：

```powershell
.\run.bat
```

无需重新选择解释器或图片；主进程沿用 TATR 环境，并自动调用独立 v6 OCR 环境。每次新结果放到 `outputs/时间/`。新电脑按 README 执行安装和模型准备，首次运行 OCR 需联网。
