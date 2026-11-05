# 默认十张测试图片

从用户已下载的 PubTables-1M **test** 图片目录复制，保留原始 JPG 字节。为便于按顺序运行，文件名前增加编号和结构类型；原文件名、尺寸、哈希、文章链接在 `manifest.json`。

选择依据是行列数量、宽高比例、合并单元格、分组标题和文字密度，十张来自十篇不同文章，另经图片总览检查。这里的“不相似”指结构和版式有差异，不是对识别效果的预先筛选。

运行项目根目录的 `run.bat` 或 `run_table_ocr.py`，默认处理这里的十张 JPG；不需要完整数据集。选择阶段用 XML 了解结构，推理阶段只读取图片，`manifest.json` 不输入模型。

数据集：https://huggingface.co/datasets/bsmock/pubtables-1m

许可：CDLA-Permissive-2.0，原文见 `DATA_LICENSE.txt`。全体样本预览在 `../docs/ten-samples-preview.png`。
