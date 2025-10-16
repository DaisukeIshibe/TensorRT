# TensorRT 8.x Precision Engine Generation

## 概要
`generate_trt8x_engine.py`がアップデートされ、FP32、FP16、INT8の精度別エンジン生成に対応しました。

## 新機能
- **精度選択**: `--precision` オプションでFP32/FP16/INT8を指定
- **一括生成**: `--all` オプションで全精度のエンジンを一括生成
- **ファイル名自動化**: 精度に応じて `model_trt8x_{precision}.trt` として保存

## 使用方法

### 個別精度指定
```bash
# FP32エンジン生成 (デフォルト)
python3 generate_trt8x_engine.py --precision fp32

# FP16エンジン生成
python3 generate_trt8x_engine.py --precision fp16

# INT8エンジン生成 (注意: 現在制限あり)
python3 generate_trt8x_engine.py --precision int8
```

### 全精度一括生成
```bash
# すべての精度でエンジン生成
python3 generate_trt8x_engine.py --all
```

### Docker実行例
```bash
# TensorRT 8.x コンテナで実行
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/tensorrt:23.03-py3 \
  python3 generate_trt8x_engine.py --all
```

## 性能比較結果

| 精度 | エンジンサイズ | サイズ削減率 | ステータス |
|------|---------------|-------------|-----------|
| **FP32** | 15.7 MB | - | ✅ 動作確認済み |
| **FP16** | 7.9 MB | **50%削減** | ✅ 動作確認済み |
| INT8 | - | - | ⚠️ 制限あり |

### サイズ削減効果
- **FP16**: FP32比で約50%のサイズ削減を実現
- **メモリ効率**: より小さなGPUメモリでの推論が可能
- **転送速度**: 軽量エンジンにより高速なモデル読み込み

## 技術的詳細

### 最適化プロファイル
すべての精度で以下の動的バッチサイズに対応：
```
最小バッチ: 1
最適バッチ: 16  
最大バッチ: 32
```

### TensorRT 8.x API対応
- `builder.build_engine()` 使用
- `max_workspace_size` 設定 (1GB)
- 動的バッチサイズ対応

### 既知の制限

#### INT8 キャリブレーション
現在の環境ではINT8キャリブレーションで以下の制限があります：
- **pycuda依存**: キャリブレーション用のGPUメモリ管理が必要
- **データ要件**: 適切な代表的データセットが必要
- **対応状況**: 技術的制約により現在は制限あり

#### 推奨事項
本番環境でのINT8使用には以下を検討：
- TensorRT 10.x環境での`tensorrt_int8_comparison.py`使用
- 適切なキャリブレーションデータセット準備
- 段階的な精度評価

## ファイル出力

### 生成されるエンジンファイル
```
model_trt8x_fp32.trt    # FP32精度エンジン (15.7 MB)
model_trt8x_fp16.trt    # FP16精度エンジン (7.9 MB)  
model_trt8x_int8.trt    # INT8精度エンジン (制限あり)
```

### キャッシュファイル
```
calibration_cache_8x.cache    # INT8キャリブレーション用
```

## C++推論での使用

生成されたエンジンは`tensorrt_inference_8x.cpp`で使用可能：

```cpp
// エンジンファイル指定
std::string engine_file = "model_trt8x_fp16.trt";  // FP16エンジン使用例
```

## トラブルシューティング

### エラー: "ONNX model not found"
```bash
# ONNXモデルを事前生成
python3 convert_to_onnx.py
```

### エラー: "TensorRT not found"
```bash
# TensorRT 8.x コンテナで実行
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/tensorrt:23.03-py3 bash
```

### 警告: "Detected subnormal FP16 values"
- FP16変換時の正常な警告
- 精度への影響は最小限
- 必要に応じてモデル再トレーニングを検討

## まとめ

✅ **成功項目**:
- FP32/FP16エンジン生成の完全サポート
- 50%のサイズ削減効果
- コマンドライン引数による柔軟な操作
- TensorRT 8.x APIの完全対応

⚠️ **注意項目**:
- INT8は現在の環境で制限あり
- 本番環境ではTensorRT 10.x推奨

この機能により、TensorRT 8.xでもFP32/FP16の精度選択が可能になり、メモリ効率と性能のバランスを取ることができます。