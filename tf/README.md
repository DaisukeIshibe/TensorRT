# TensorRT Model Verification Pipeline

このプロジェクトは、TensorFlowで作成したモデルがTensorRTで実行した時と同じ結果を得ることを確認するための包括的な検証パイプラインです。

## 概要

以下のステップを自動化して実行します：

1. **Kerasでcifar10の画像分類CNNモデルを作成** (SavedModel形式で保存)
2. **SavedModelをONNX形式に変換**
3. **ONNXモデルをTensorRT形式に変換**
4. **Python版TensorRTで推論**し、SavedModelと結果を比較
5. **C++版TensorRTで推論**し、Python版と結果を比較

## 必要な環境

### Dockerイメージ
- `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` (ステップ1-2用)
- `nvcr.io/nvidia/tensorrt:25.06-py3` (ステップ3-5用)

### ハードウェア
- NVIDIA GPU (CUDA対応)
- Docker with NVIDIA Container Toolkit

## ファイル構成

```
cifar10.py                 - CIFAR-10 CNNモデルの学習・保存
convert_to_onnx.py        - SavedModel → ONNX変換
convert_to_tensorrt.py    - ONNX → TensorRT変換
compare_models.py         - 全モデル形式の推論結果比較 (Python)
tensorrt_inference.cpp    - TensorRT推論プログラム (C++)
CMakeLists.txt           - C++プログラムのビルド設定
run_tensorflow_steps.sh   - TensorFlow関連ステップの実行
run_tensorrt_steps.sh    - TensorRT関連ステップの実行
run_complete_pipeline.sh - 全パイプラインの実行
analyze_results.py       - 最終結果の詳細分析
```

## 使用方法

### 方法1: 完全自動実行

```bash
# 全ステップを自動実行
./run_complete_pipeline.sh
```

### 方法2: ステップ別実行

```bash
# ステップ1-2: TensorFlow関連 (モデル学習・ONNX変換)
./run_tensorflow_steps.sh

# ステップ3-5: TensorRT関連 (TensorRT変換・Python/C++推論)
./run_tensorrt_steps.sh
```

### 方法3: 手動実行

#### TensorFlowコンテナでの実行:
```bash
# Docker イメージを起動
docker run --rm -it --gpus all --ipc=host \
  -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/tensorflow:25.02-tf2-py3 bash

# コンテナ内で実行
pip install tf2onnx onnxruntime-gpu tabulate
python cifar10.py              # モデル学習
python convert_to_onnx.py      # ONNX変換
```

#### TensorRTコンテナでの実行:
```bash
# Docker イメージを起動
docker run --rm -it --gpus all --ipc=host \
  -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/tensorrt:25.06-py3 bash

# コンテナ内で実行
pip install numpy tabulate onnxruntime-gpu
python convert_to_tensorrt.py  # TensorRT変換
python compare_models.py       # Python推論・比較

# C++プログラムのビルド・実行
mkdir build && cd build
cmake .. && make -j$(nproc)
./tensorrt_inference ../model.trt ../test_samples.npy ../cpp_results.csv
```

## 結果の分析

パイプライン実行後、以下のコマンドで詳細分析を実行できます：

```bash
python analyze_results.py
```

これにより以下が出力されます：
- 各モデル形式の分類精度
- モデル間の予測値の一致性
- サンプル別の詳細比較
- 分析サマリーファイル

## 生成されるファイル

### モデルファイル
- `cifar10_vgg_model/` - TensorFlow SavedModel
- `model.onnx` - ONNX形式モデル
- `model.trt` - TensorRT エンジンファイル

### 予測結果
- `test_samples.npy` - テスト用画像データ
- `test_labels.npy` - 正解ラベル
- `*_predictions*.npy` - 各モデルの予測結果
- `cpp_tensorrt_results.csv` - C++版の予測結果

### 分析結果
- `analysis_summary.txt` - 分析サマリー

## 期待される結果

正常に動作する場合：
1. 全てのモデル形式で高い分類精度（通常70%以上）
2. SavedModel、ONNX、TensorRT間での予測値の高い一致性
3. Python版とC++版TensorRTの結果の一致

## トラブルシューティング

### Dockerイメージが見つからない場合
```bash
docker pull nvcr.io/nvidia/tensorflow:25.02-tf2-py3
docker pull nvcr.io/nvidia/tensorrt:25.06-py3
```

### GPU関連エラーの場合
- NVIDIA Container Toolkitがインストールされているか確認
- `nvidia-docker` または `docker --gpus all` が使用可能か確認

### メモリ不足エラーの場合
- 学習エポック数を減らす (cifar10.py内のepochs=5をより小さく)
- バッチサイズを減らす

### ビルドエラー (C++)の場合
- TensorRTのインストールパスを確認
- CMakeLists.txtのパス設定を環境に合わせて調整

## カスタマイズ

### モデル設定の変更
`cifar10.py`でモデルアーキテクチャや学習パラメータを変更可能

### 精度設定の変更
`convert_to_tensorrt.py`でTensorRTの精度(fp32/fp16/int8)を変更可能

### 比較許容値の調整
`compare_models.py`と`analyze_results.py`で一致性判定の許容値を調整可能

## ライセンス

このプロジェクトは検証・教育目的で作成されています。