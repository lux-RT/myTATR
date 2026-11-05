# OCR 接入说明

## 修改范围

这次是外部集成，不是在 TATR 网络中增加可训练层。此前已经能用 Transformers 加载 TATR 并输出结构框；本次把这段能力做成持久的命令行入口，并新增 OCR 进程、数据格式适配、单元格融合和导出。

原有项目文件、模型权重和数据集没有修改。当前新仓库原本为空，因此没有覆盖旧分支代码；实现完成后提交到同一个 `main` 分支。

## 运行时的数据如何流动

1. `run_table_ocr.py` 检查图片和本地模型，并拒绝输出文件名冲突。
2. 主进程用 `subprocess.run(..., check=True)` 调用 `.venv-ocr` 中的 `ocr_worker.py`。两个环境之间只交换 UTF-8 JSON，避免 OCR 依赖升级现有 PyTorch/Transformers。
3. `ocr_worker.py` 在 PIL 中统一 EXIF 方向，默认放大 2 倍后，用 RapidOCR 检测和识别；关闭 180 度方向分类器，适合当前朝向正常的数据集样本。
4. `normalize_lines()` 将 OCR 四边形转换为 `[xmin,ymin,xmax,ymax]`。使用放大后真实宽高与原图宽高的比值分别还原两个方向，避免非整数放大的舍入误差。
5. 主进程加载 `TableTransformerForObjectDetection`，保持原有本地权重和 `AutoImageProcessor` 预处理，得到原图坐标的行列、表头、合并区域。
6. `fuse()` 将 Transformers 字符串标签转换为微软 `postprocess.objects_to_cells()` 需要的类别索引，复用对齐、去重和合并单元格处理。这里先传空 token 列表构建结构，防止官方“删除无文字行列”的规则把 OCR 漏字误当成空结构。
7. 再调用 `slot_into_containers()`，采用 `overlap_threshold=0.5`、`unique_assignment=True`、`forced_assignment=False`，把每个 OCR token 分配到覆盖其面积最多的单元格；同格文字按几何位置组成多行文本。
8. 无法分配的文字单独保存，通常包括标题、脚注；同时明显覆盖多个单元格的 OCR 框标为歧义，不用置信度掩盖这种情况。
9. 导出单元格 JSON、支持跨行跨列的 HTML、CSV 和坐标标注图。CSV 使用 `newline=""`，防止 Windows 把 CRLF 再次转换而出现双倍行数。

## 为什么是 RapidOCR 而不是直接安装 PaddleOCR

讨论阶段建议过 PaddleOCR。实际实现采用 RapidOCR-ONNXRuntime 1.4.4，它打包 PP-OCRv4 检测/识别 ONNX 模型，便于 Windows CPU 安装及离线复跑。它与 TATR 通过统一的 token JSON 对接；以后切换 PaddleOCR、新版 RapidOCR 或英文专用 OCR 时，可以保留融合、导出和 TATR 推理部分。

当前 OCR 输出是行级文字框。它可能把相邻列的文字看成同一行；这个问题需要进一步引入词/字符框，或对歧义单元格裁图重识别。此版本保留并报告问题，未通过人工改写预测文字制造正确结果。

## 如何追溯第三方代码

`third_party/tatr/postprocess.py` 来自官方仓库固定提交，文件内容未修改。`provenance.json` 记录 URL、提交和 SHA-256；官方 MIT 许可原文保留。TATR 模型也使用固定 revision，`prepare_model.py` 提供文件校验。依赖环境分别由两个 `requirements-*-lock.txt` 记录。
