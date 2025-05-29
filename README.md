# Euler-a-ISP  
### 　(Inverse Scattering Problem Enhanced Euler a Sampler)

## 概要
　**Euler-a-ISP** は、Stable Diffusion のサンプリングプロセスに **逆散乱問題 (Inverse Scattering Problem)** を組み込んだ **カスタム Euler a サンプラー** です。従来の `Euler a` サンプラーに比べ画像生成時の確率的な変化を制御し **推論の進行を調整** することができます。

## 特徴
(カスタムパラメータ以降にさらに詳しい特徴等を記します)  
- **逆散乱問題 (Inverse Scattering Problem)** を活用した Ancestral Sampling の補正  
- **進行の安定性向上**: ノイズ抑制によりスムーズな推論を実現  
- **調整可能なパラメータ (`alpha`, `beta`, `omega`)** によるカスタマイズ  
- **`Euler a` の特性を維持しつつ精度と滑らかさを強化(デフォルト値)**  
- **Stable Diffusion WebUI (A1111, Forge, ReForge)** に対応

## インストール方法
```bash
git clone https://github.com/muooon/euler-a-isp.git  
```
使い方  
1. 	フォルダを Stable Diffusion WebUI の extensions に配置  
2. 	WebUI を起動し euler-a-isp をサンプラーリストから選択  
3. 	好みで `alpha`, `beta`, `omega` を調整し推論の進行をカスタマイズ可 (デフォルト／`a:0.1`,`b:0.02`,`omega:5`)  
4. 	画像生成を試し補正の効果を確認  

カスタムパラメータ (参考値)  

| パラメータ | 説明 | 推奨値 |  
|------------|-------------------|------------|  
| **alpha**  | 逆散乱問題の影響係数 | 0.1 ~ 1.0  |  
| **beta**   | ノイズ除去率         | 0.01 ~ 0.1 |  
| **omega**  | 波動周期           | 1 ~ 10     |  
- Alpha (α)：前ステップ値の影響、大きいほど過去値を強く維持し小さいほど新値の影響を受ける  
- Beta (β)：ノイズの振幅やスケールを調整 / 高値はランダム性が強くし大きな変動を生む  
- Omega (ω)：補正値の適応度を決める係数で安定性を決定する変化のスムーズさに関係する  

その他の調整とサンプラーの特殊さ  

- モデル自体の持つ特徴をよく表現するようになります  
- スケジューラを変えるだけでも アニメ感 / リアル感 / 書き込み量 変化させます  
- ３要素を任意値にするだけで多種多様な変化を起こせます  
- 任意のスケジューラと３要素への任意のパラメータにより多種多様なモデルを変化させます  
- 参考値を超えての使用もできます(以下の進行比較をご覧ください)  

サンプラー進行の比較  
![image01](https://github.com/muooon/euler-a-isp/blob/main/ISP-EulerA-GRAF-02GEN-hikaku02.png)  
euler-a-isp の推論進行の視覚化  
![image01](https://github.com/muooon/euler-a-isp/blob/main/euler-a-isp-iroiro.png)  
Euler a / ISP の進行 (スケジューラ適用)  
![image01](https://github.com/muooon/euler-a-isp/blob/main/euler-a-isp-iroiro02.png)  

推論の視覚化  
　推論の進行を視覚化するには以下の Python コードを実行してください  
```bash
import numpy as np
import matplotlib.pyplot as plt

# 日本語フォント設定 (Windows用)
plt.rcParams["font.family"] = "MS Gothic"

# ステップ数
steps = 50
t = np.linspace(0, steps, steps)

# Euler a の進行
euler_a_values = np.exp(-0.02 * t) * (1 + np.sin(5 * t))

# ユーザー入力
print("Euler a ISP のパラメータを入力してください:")
alpha = float(input("Alpha 値を入力: "))
beta = float(input("Beta 値を入力: "))
omega = float(input("Omega 値を入力: "))

# Euler a ISP の進行（ユーザー入力値を適用）
euler_isp_values = np.exp(-beta * t) * (1 + alpha * np.sin(omega * t))

# グラフの描画
plt.figure(figsize=(10, 6))

# Euler a をプロット
plt.plot(t, euler_a_values, linestyle="-", label="Euler a", color="blue")

# Euler a ISP をプロット（太い線にする）
plt.plot(t, euler_isp_values, linestyle="-", label=f"Euler a ISP (α={alpha}, β={beta}, Ω={omega})", color="red", linewidth=2.5)

plt.xlabel("ステップ数")
plt.ylabel("推論値")
plt.title("Euler a と Euler a ISP の比較グラフ")
plt.legend()
plt.grid()
plt.show()
```
### AIによる数学的解説
　Euler A における補正の計算式：
\[
x_{n+1} = x_n + C \cdot (\eta \cdot \sigma)
\]
- **前ステップの値** (\( x_n \))
- **ランダムノイズ** (\( \eta \))
- **ノイズのスケール** (\( \sigma \))
- **補正値** (\( C \))

　ISP における補正の計算式：
\[
x_{n+1} = x_n + C(x_n) \cdot (\eta \cdot \sigma)
\]

　この補正計算において、3要素（\( \alpha, \beta, \omega \)）は以下のように関係します：
\[
x_{n+1} = \alpha \cdot x_n + \omega \cdot C(x_n) \cdot (\beta \cdot \eta \cdot \sigma)
\]
- **前の値の影響度** (\( \alpha \)) ：前の値をどの程度保持するかを決定する係数
- **ノイズスケール調整** (\( \beta \)) ：ノイズの大きさを調整しランダム性の強さを決める係数
- **補正値の適用度** (\( \omega \)) ：補正値がどの程度適用されるかを決定する係数  

ライセンス  
　このプロジェクトは Apache License 2.0 のもと提供されています  
　詳細なライセンス情報については以下をご参照ください  
　Apache License 2.0
