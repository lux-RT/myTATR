# 千图测试完整逐表结果

这是 2026-11-11 完成的两套实验导出结果，来自同一批随机固定的 1000 张 PubTables-1M 测试表格，随机种子为 42。实验条件、指标解释和分组结果见 [千图双版本测试报告](../README.md)。

## 新版：包含 PNG 结构图

2026-11-14 已给两套结果各补齐 1000 张 PNG 并重新打包。因为每包超过 GitHub 普通 Git 文件的 100 MiB 限制，完整 ZIP 放在 [本仓库的 Releases](https://github.com/lux-RT/myTATR/releases/tag/benchmark-1000-png-20261114)，可直接下载：

| 下载 | 文件大小（十进制 MB） | 内容 |
|---|---:|---|
| [官方基线 1000 张，含 PNG](https://github.com/lux-RT/myTATR/releases/download/benchmark-1000-png-20261114/benchmark-1000-official-with-png.zip) | 108.48 MB | `official/` 下各 1000 份 HTML、CSV、JSON、PNG |
| [接入 OCR 1000 张，含 PNG](https://github.com/lux-RT/myTATR/releases/download/benchmark-1000-png-20261114/benchmark-1000-ocr-with-png.zip) | 107.15 MB | `current/` 下各 1000 份 HTML、CSV、JSON、PNG |

每包共 **4007 个文件**：四种逐表结果各 1000 份，另含原有 7 个指标、名单、说明和许可文件。解压后，浏览器打开 HTML 看重建表格；打开 `<表格ID>_overlay.png` 看原图上的结构与文字框；CSV 可用 Excel 打开；JSON 保留完整坐标、文字和指标。

蓝框表示普通单元格，紫框表示跨行或跨列单元格，绿框表示词框。官方基线绿框来自官方词信息，当前系统绿框来自 PP-OCRv6 small。PNG 直接使用原图及已保存坐标绘制，没有重新推理或改写指标；若结果有重叠单元格，图中如实保留。

两个新版 ZIP 均通过 CRC、固定名单覆盖及 PNG 与本地文件逐字节对比；除更新 `说明.txt` 和 `SOURCE.txt` 外，旧包中的全部文件内容保持一致。校验见 [SHA256SUMS-with-png.txt](SHA256SUMS-with-png.txt)，验证摘要见 [png-package-verification.json](png-package-verification.json)。Releases 也附同一份校验文件，名为 `SHA256SUMS.txt`。

新版包不含模型权重、推理缓存或单独的原始 JPG；PNG 本身以原表格图片为底图。许可和来源说明包含在每个 ZIP 中。两套包按相同表格 ID 命名，方便比较。

## 历史版本：不含 PNG

| 下载 | 文件大小 | 内容 |
|---|---:|---|
| [官方基线 1000 张结果](官方基线-1000张结果.zip) | 7.30 MB | `official/` 下各 1000 份 HTML、CSV、JSON |
| [接入 OCR 1000 张结果](接入OCR-1000张结果.zip) | 3.42 MB | `current/` 下各 1000 份 HTML、CSV、JSON |

在 GitHub 打开 ZIP 文件后，点击 **Download raw file** 下载并解压。HTML 可用浏览器打开，CSV 可用 Excel 打开，JSON 保存完整预测单元格和对应指标。两套文件按相同的表格 ID 命名，便于逐表比较。

每个压缩包另含以下 7 个文件，共 3007 个文件：

- `metrics.csv`：1000 张表的逐表指标。
- `summary.json`：全部样本及简单表、复杂表的统计。
- `manifest.json`：固定抽样名单、顺序及原图和标注的 SHA-256；已省略本机绝对路径。
- `export-diagnostics.json`：重叠单元格等导出歧义的案例名单。
- `说明.txt`：查看方法和结果解释。
- `DATA_LICENSE.txt`、`SOURCE.txt`：数据许可和实验来源。

官方基线使用官方词信息，当前系统使用 PP-OCRv6 small。两套流程还存在预处理和后处理差异，不能将指标差值全部归因于 OCR。官方基线和当前系统分别有 25 张、27 张结果包含重叠单元格；HTML/CSV 无法完整表达这种歧义，JSON 保留全部单元格。正式指标按导出前的原始单元格计算，未通过修改预测提高分数。

压缩包包含完整的逐表导出结果，不包含原始图片、模型权重或推理缓存。原始图片可按固定名单从 [PubTables-1M](https://huggingface.co/datasets/bsmock/pubtables-1m) 获取；重新运行的步骤见 [千图测试运行说明](../../千图测试运行说明.md)。

发布包在本地原始结果包的基础上仅增加许可与来源说明，原有 3005 个文件逐字节保持一致。两个 ZIP 均通过 CRC 校验、固定名单覆盖检查及逐表指标数量检查；下载后可对照 [SHA256SUMS.txt](SHA256SUMS.txt) 校验文件。
