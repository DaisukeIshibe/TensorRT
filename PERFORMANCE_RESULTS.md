# 🚀 性能比較結果：SavedModel vs ONNX vs TensorRT

## 📊 総合性能比較

| モデル形式 | 推論時間 (100サンプル) | スループット (samples/sec) | ロード時間 | 備考 |
|------------|-------------------------|---------------------------|------------|------|
| **SavedModel** | 0.180s ± 0.240s | 555.6 | 2.677s | TensorFlow |
| **ONNX** | 0.012s ± 0.005s | 8,556.7 | 0.153s | 最高速 |
| **TensorRT Batch** | 0.0118s ± 0.0002s | **8,483.2** | 0.121s | **最適化** |
| **TensorRT Single** | 0.1199s (scaled) | 834.3 | 0.121s | 単一処理 |

## 🏆 性能向上率

### 1. ONNX vs SavedModel
- **15.4倍高速** (555.6 → 8,556.7 samples/sec)
- ロード時間17.8倍改善 (2.677s → 0.153s)

### 2. TensorRT Batch vs SavedModel
- **15.3倍高速** (555.6 → 8,483.2 samples/sec)
- ロード時間22.1倍改善 (2.677s → 0.121s)

### 3. TensorRT Batch vs Single
- **10.2倍高速** (834.3 → 8,483.2 samples/sec)
- GPU転送回数96%削減 (100回 → 4回)

## 📈 バッチ処理の効果

### GPU転送最適化
```
単一処理: 100サンプル = 100回のGPU転送
バッチ処理: 100サンプル = 4回のGPU転送 (32サンプル/バッチ)
削減率: 96%
```

### メモリ効率
- バッチサイズ32で最適化
- 動的バッチサイズ対応 (min=1, opt=16, max=32)
- スマートポインタによる安全なメモリ管理

## 🔧 技術仕様

### TensorRT設定
- **バージョン**: 10.11.0.33
- **最適化プロファイル**: 
  - Minimum batch size: 1
  - Optimal batch size: 16  
  - Maximum batch size: 32
- **精度**: FP32

### 測定環境
- **Docker**: nvidia/tensorrt:25.06-py3
- **GPU**: CUDA対応GPU
- **測定回数**: 各5回の平均値

## 📝 結論

1. **ONNX**: 最も高いスループット (8,556.7 samples/sec)
2. **TensorRT Batch**: ONNXに匹敵する高速性能 (8,483.2 samples/sec) + GPU最適化
3. **バッチ処理**: 単一処理より10.2倍高速、GPU転送96%削減
4. **SavedModel**: 基準値 (555.6 samples/sec)

### 推奨用途
- **高速推論が必要**: ONNX or TensorRT Batch
- **GPU最適化重視**: TensorRT Batch
- **開発・プロトタイプ**: SavedModel
- **本番環境**: TensorRT Batch (メモリ効率+高速性)

---
*測定日時: 2025年1月14日*
*環境: Docker + NVIDIA TensorRT 25.06*