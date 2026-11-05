# TATR + OCR 本地实测（2026-11-05）

四张图片均取自已下载的 **PubTables-1M 测试集**，未使用 XML 或文字真值参与推理。原图的四个小样本放在 `input/`，用于重现本报告；完整数据集和权重不上传。

打开 [`index.html`](index.html) 查看原图、标注图及重建表格；[`comparison.png`](comparison.png) 是四张图的原图/坐标框总览。每张图另有 HTML、CSV、完整预测 JSON。

## 实测环境

- Windows x64 / Python 3.12.3。
- TATR：PyTorch `2.8.0+cpu`、Transformers `4.57.1`，本地原始预训练结构模型的 Transformers 权重，未训练或修改。
- OCR：RapidOCR-ONNXRuntime `1.4.4`、ONNX Runtime `1.23.2`，包内 PP-OCRv4 检测/识别模型，默认放大 2 倍。
- 两个独立虚拟环境；CPU 各使用 4 个线程；TATR 结构阈值 0.50，文字与单元格匹配要求至少覆盖文字框面积的 50%。
- 最终这轮设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 后完整运行退出码为 0；OCR 使用随包安装的本地模型，不发起模型下载。

## 结果

| 图片 | 行 × 列 | 填有文字的单元格 / 总单元格 | 分配入表的 OCR 框 / 所有 OCR 框 | 分阶段耗时合计 |
|---|---:|---:|---:|---:|
| PMC1262774_table_0 | 5 × 4 | 18 / 20 | 41 / 43 | 6.05 秒 |
| PMC1064078_table_0 | 9 × 6 | 48 / 50 | 48 / 49 | 3.61 秒 |
| PMC1064082_table_1 | 24 × 4 | 65 / 74 | 65 / 67 | 4.30 秒 |
| PMC1064879_table_0 | 33 × 5 | 165 / 165 | 165 / 168 | 8.18 秒 |

耗时为 OCR 图片处理 + TATR 预处理/推理/解码 + 单元格融合之和，不含 Python/依赖冷启动、进程启动、模型初始化和文件导出。完整分阶段数据在 [`summary.json`](summary.json)。

合计 327 个 OCR 行框，319 个分配入表，8 个未分配框是此次识别出的标题或脚注。**框被成功分配不等于框里的文字识别正确，也不保证每个框都分配到正确单元格。**

独立读取 XML 后，四张图的预测行列数均与标注一致。合并表头、跨行标题、分组行标题已经输出为合并单元格；CSV 不支持合并，文字放在合并范围的左上角。

## 实际发现的错误

1. `PMC1262774_table_0` 有两个框被标记为几何歧义。其中一条 OCR 把最后一行的第 3 列和第 4 列文字识别成同一个长行，整行被放进第 3 列，第 4 列因此为空。这是本次可见的文字归属错误；另一框靠近行边界。
2. `PMC1064082_table_1` 的 `≥50` 被识别为 `250`，`16(33%)` 被识别为 `16(339)`。
3. `PMC1064078_table_0` 的一处 `Controls (%)` 被识别成 `Controis(9s)`；部分英文单词间空格丢失。
4. 第三张表继承 TATR 原始输出的一处低分合并单元格预测（约 0.505），并不是 OCR 引入的新结构判断。

以上结果未经人工修改。本次完成了 OCR 与 TATR 的接入及样本测试，尚未进行文字准确率、TEDS 或 GriTS 的完整评估。论文表格截图也不能代表倾斜、反光或透视明显的真实手机照片效果。

## 自动验证

- 14 项单元测试通过，覆盖缩放坐标、左右列文字分配、空白行保留、空 OCR、表外文字、合并表头、HTML 转义、CSV 多行内容和 Windows 换行、输入名冲突。
- 两个环境 `pip check` 均通过。
- 四份 CSV 解析尺寸分别为 5×4、9×6、24×4、33×5，没有 Windows CRCRLF 导致的额外空行。
- 四份 JSON 的单元格恰好覆盖整个行列网格，无重复占格；所有 OCR token 都能追溯为已分配或未分配。
- 官方后处理源码及许可证哈希与固定来源一致；本地 TATR 三个文件的哈希与下载时记录一致。
- 独立代码审查发现的问题已修复并复核。

## 复现

先按仓库 README 安装环境和准备模型，然后在仓库根目录执行：

```powershell
$images = (Get-ChildItem -LiteralPath 'examples/pubtables-smoke/input' -Filter '*.jpg').FullName
& '.\.venv-tatr\Scripts\python.exe' run_table_ocr.py --images $images --out outputs/reproduce
```

在原电脑上，将解释器换成 `..\PubTables-1M复现\.venv-download\Scripts\python.exe` 即可。图片处理先后顺序不同可能使计时略有变化。

## 样本来源

数据集：[bsmock/pubtables-1m](https://huggingface.co/datasets/bsmock/pubtables-1m)，CDLA-Permissive-2.0，许可原文见 [`DATA_LICENSE.txt`](DATA_LICENSE.txt)。文件名、原始文章链接和原图 SHA-256 在 [`input_provenance.json`](input_provenance.json)。

- [PMC1262774](https://pmc.ncbi.nlm.nih.gov/articles/PMC1262774/)
- [PMC1064078](https://pmc.ncbi.nlm.nih.gov/articles/PMC1064078/)
- [PMC1064082](https://pmc.ncbi.nlm.nih.gov/articles/PMC1064082/)
- [PMC1064879](https://pmc.ncbi.nlm.nih.gov/articles/PMC1064879/)
