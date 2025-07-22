# lstm_sample
machine learning lstm sample

Pythonのバージョンは、3.11.4です。
ターゲットはApple SiliconのMacです。

(0) インストール
% pip install -r requirement.txt

すればライブラリ入るはず。

(1) 学習

% python collect.py <動画ファイル>　<ラベル名>

これを準備した動画する分だけやる。

(2) LSTMでの機械学習/GRUでの機械学習


python train.py で学習
dataディレクトリ下にあるラベルごとのデータファイルを読み込んで学習する。

train.pyのmodel.fitでepoch数を指定する。
```
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
```
この例では、3０epochである。学習を30回くりかえすことになる。

GRUアルゴリズムでも学習できる。それほど時間は変わらない。
```
python train_gru.py
```
でOK。

```
(tfenv) hayakawa@EiichinoMac-mini lstm_sample % python train.py
[INFO] Loading data from 'data/'...
[INFO] Loaded 7 samples across 1 labels.
[INFO] Encoded labels and saved to labels.npy
[INFO] Training model...
Epoch 1/30
1/1 [==============================] - 4s 4s/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 2/30
1/1 [==============================] - 0s 31ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 3/30
1/1 [==============================] - 0s 28ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 4/30
1/1 [==============================] - 0s 27ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 5/30
1/1 [==============================] - 0s 25ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 6/30
1/1 [==============================] - 0s 26ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 7/30
1/1 [==============================] - 0s 23ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 8/30
1/1 [==============================] - 0s 28ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 9/30
1/1 [==============================] - 0s 28ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 10/30
1/1 [==============================] - 0s 24ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 11/30
1/1 [==============================] - 0s 24ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 12/30
1/1 [==============================] - 0s 24ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 13/30
1/1 [==============================] - 0s 27ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 14/30
1/1 [==============================] - 0s 24ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 15/30
1/1 [==============================] - 0s 26ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 16/30
1/1 [==============================] - 0s 23ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 17/30
1/1 [==============================] - 0s 28ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 18/30
1/1 [==============================] - 0s 28ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 19/30
1/1 [==============================] - 0s 29ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 20/30
1/1 [==============================] - 0s 24ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 21/30
1/1 [==============================] - 0s 23ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 22/30
1/1 [==============================] - 0s 24ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 23/30
1/1 [==============================] - 0s 28ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 24/30
1/1 [==============================] - 0s 29ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 25/30
1/1 [==============================] - 0s 24ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 26/30
1/1 [==============================] - 0s 33ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 27/30
1/1 [==============================] - 0s 26ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 28/30
1/1 [==============================] - 0s 24ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 29/30
1/1 [==============================] - 0s 29ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 30/30
1/1 [==============================] - 0s 28ms/step - loss: 0.0000e+00 - accuracy: 0.0000e+00 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
/Users/hayakawa/tfenv/lib/python3.11/site-packages/keras/src/engine/training.py:3000: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(
[INFO] Model trained and saved as gesture_lstm.h5
```
これが出ればモデルの生成が完了。

(3)　検証

% python recoginize_gesture.py <検証用動画>

これで左上に適切なラベルがつけば認識が成功しているはず。

GRUモデルにするときは、モデルファイルを修正する。
```
model = tf.keras.models.load_model("gesture_lstm.h5")  #こちらがモデル
```
これを、
```
model = tf.keras.models.load_model("gesture_gru.h5")  #こちらがモデル
```
これに修正

(4) 2Dでの検証（LSTMだけ）
ポーズ認識だと２次元でも結構いけるのではと思います（3Dだと情報が多すぎるので）。

```
make allclean
python collect.py dance.mp4 dance
python train2d.py ← ここが違う
python recog2d.py dance.mp4
```

これでやると100%になるみたいね。学習させるときには、make allcleanとうってファイルをクリアしておくこと。



