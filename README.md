# Euler-a-ISP  
# (Inverse Scattering Problem Enhanced Euler a Sampler)

## 概要
**Euler-a-ISP** は、Stable Diffusion のサンプリングプロセスに **逆散乱問題 (Inverse Scattering Problem)** を組み込んだ **カスタム Euler a サンプラー** です。従来の `Euler a` サンプラーに比べて、画像生成時の確率的な変化を制御し、**推論の進行を滑らかに調整** することができます。

## 特徴
- **逆散乱問題 (Inverse Scattering Problem)** を活用した Ancestral Sampling の補正  
- **進行の安定性向上**: ノイズ抑制によりスムーズな推論を実現  
- **調整可能なパラメータ (`alpha`, `beta`, `omega`)** によるカスタマイズ  
- **`Euler a` の特性を維持しつつ、精度と滑らかさを強化**  
- **Stable Diffusion WebUI (A1111, Forge, ReForge)** に対応

## インストール方法
```bash
git clone https://github.com/muooon/euler-a-isp.git
cd euler-a-isp
```
使い方  
1. 	フォルダを Stable Diffusion WebUI の extensions に配置  
2. 	WebUI を起動し euler-a-isp をサンプラーリストから選択  
3. 	好みで `alpha`, `beta`, `omega` を調整し推論の進行をカスタマイズ可 (デフォルト／`a:01`,`b:0.02`,`omega:5`)  
4. 	画像生成を試し補正の効果を確認  

カスタムパラメータ  

`| パラメータ | 説明 | 推奨値 |`   
`| alpha | 逆散乱問題の影響係数 | 0.1 ~ 1.0 |`   
`| beta | ノイズ除去率 | 0.01 ~ 0.1 |`   
`| omega | 波動周期 | 1 ~ 10 |`   

推論の視覚化  
推論の進行を視覚化するには、以下の Python コードを実行してください。
```bash
import numpy as np
import matplotlib.pyplot as plt

steps = 50
tau = np.linspace(0, steps, steps)
euler_values = np.exp(-0.02 * tau)
euler_isp_values = np.exp(-0.02 * tau) * (1 + 0.1 * np.sin(5 * tau))

plt.plot(tau, euler_values, label="Euler", color="black")
plt.plot(tau, euler_isp_values, label="Euler-a-ISP", color="red", linestyle="--")
plt.xlabel("ステップ数")
plt.ylabel("推論の進行")
plt.title("Euler vs Euler-a-ISP の比較")
plt.legend()
plt.grid()
plt.show()
```
ライセンス  
このプロジェクトは Apache License 2.0 のもと提供されています。
詳細なライセンス情報については、以下をご参照ください。
Apache License 2.0
