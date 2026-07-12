# rmv-watermarks

轻量、可插拔的 AI 可见水印去除库。默认支持 Google Gemini / Nano Banana 右下角星形水印，基于数学精确的反向 Alpha 混合算法，CPU 即可运行。

## 特性

- **通用插件架构**：通过 `WatermarkProvider` 扩展其他水印类型
- **零 GPU 依赖**：仅需 NumPy、OpenCV、Pillow
- **统一 API**：支持文件路径、字节流、NumPy 数组
- **可逆验证**：合成/去除 round-trip 测试保证算法正确性

## 安装

```bash
pip install -e ".[dev]"
```

## CLI

```bash
# 去除水印
rmv-watermarks photo.png -o clean.png

# 仅检测
rmv-watermarks photo.png --detect-only --json

# 强制去除（跳过检测）
rmv-watermarks photo.png -o clean.png --force
```

## Python API

```python
import rmv_watermarks as rmv

# 自动检测并去除
result = rmv.remove("watermarked.png", "clean.png")
print(result.removed, result.confidence)

# 仅检测
det = rmv.detect("image.png")
print(det.detected, det.region)

# 查看已注册 provider
print(rmv.list_providers())  # ['gemini']
```

## 算法

Gemini 使用标准 Alpha 合成：

```
watermarked = α × logo + (1 - α) × original
```

去除时反解：

```
original = (watermarked - α × logo) / (1 - α)
```

Alpha 图从纯背景捕获图预计算，实现无损恢复（8-bit 量化除外）。

## 开发

```bash
pytest
ruff check src tests
```

## 许可

MIT
