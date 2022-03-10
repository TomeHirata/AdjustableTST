# Code

## 1. データセットの準備

data/README.mdの指示に従い、YelpとAmazonのトレーニング、バリデーション、テストデータセットを作成してください。このREADMEの残りの部分では、Yelpを例として、どのように実行するかを説明します。

データセット作成後、加工されたYelpレビューデータセットを含むフォルダ `../data/dataset/yelp/processed/style_transfer/` が作成されるはずです。

## 2. 属性判別器の学習

```bash
mkdir ../data/dataset/yelp/processed/style_transfer/classifier_training
cd ../data/dataset/yelp/processed/style_transfer/
```

### fastText感情判別器の学習用データの作成

```bash
awk -F "\t" '{print $1" __label__"$4}' train.fader.with_cat.proc.40000 > classifier_training/train.binary_sentiment.40000.txt
sed -i '/__label__0/d' classifier_training/train.binary_sentiment.40000.txt

awk -F "\t" '{print $1" __label__"$4}' valid.fader.with_cat.proc.40000 > valid.binary_sentiment.40000.txt
sed -i '/__label__0/d' valid.binary_sentiment.40000.txt
```

### fastText感情判別器の学習

```bash
fasttext supervised -wordNgrams 4 -minn 3 -maxn 3 -input classifier_training/train.binary_sentiment.40000.txt -output classifier_training/fasttext.binary_sentiment.40000

fasttext test classifier_training/fasttext.binary_sentiment.40000.bin classifier_training/valid.binary_sentiment.40000.txt
→判別器の精度が表示されるはずです
```

## 3. データセットのバイナリ化

データを素早く読み込むために、データを `.pth` ファイルにバイナリ化します。

```bash
bash code/binarize_yelp_data.sh
```

以下のファイルが作成されているはずです。

```bash
data/dataset/yelp/processed/style_transfer/train.fader.with_cat.proc.<bpe_codes>.pth
data/dataset/yelp/processed/style_transfer/valid.fader.with_cat.proc.<bpe_codes>.pth
data/dataset/yelp/processed/style_transfer/test.fader.with_cat.proc.<bpe_codes>.pth
```

## 4. 属性変換モデルの学習

モデルの学習には、`code/main-parallel.py`スクリプトを用いる。

### 感情変換

```bash
mkdir models/style_transfer
python main-parallel.py --exp_name test \
    --dump_path models/style_transfer \
    --mono_dataset ../data/dataset/yelp/processed/style_transfer/train.fader.with_cat.proc.40000.pth,../data/dataset/yelp/processed/style_transfer/valid.fader.with_cat.proc.40000.pth,../data/dataset/yelp/processed/style_transfer/test.fader.with_cat.proc.40000.pth \
    --attributes binary_sentiment \
    --n_mono -1 \
    --lambda_ae 1.0 \
    --lambda_bt 0.6 \
    --train_ae true \
    --train_bt true \
    --eval_ftt_clf binary_sentiment:../data/dataset/yelp/processed/style_transfer/fasttext.binary_sentiment.40000.bin \
    --bleu_script_path ../data/mosesdecoder/scripts/generic/multi-bleu.perl \
    --balanced_train true
```

このフォルダには、トレーニングログファイル、モデルのチェックポイント、各イテレーションでの検証/テストデータセットでの生成結果を表示します。

ログファイルには、検証やテストデータセットでの分類器の精度とself-BLEU Scoreが含まれます。
