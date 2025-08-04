# Autocorrelation Function (ACF)
<img width="1830" height="842" alt="image" src="https://github.com/user-attachments/assets/cbffdc14-f333-42d7-aed9-f501963bb0a7" />
<img width="1796" height="914" alt="image" src="https://github.com/user-attachments/assets/95ae7dcc-a9b4-417e-95a3-16adba2fb0eb" />

```
# 导入所需库
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

# 加载 CO₂ 每周数据
co2_weekly = sm.datasets.co2.load_pandas().data

# 重采样为每月平均值
co2_monthly = co2_weekly['co2'].resample('M').mean()

# 用插值法填补缺失值
co2_series = co2_monthly.interpolate()

# 绘制 ACF 图
plt.figure(figsize=(8, 4))
plot_acf(co2_series, lags=40)
plt.title('Autocorrelation (ACF) of CO₂ Series')
plt.show()

```
<img width="608" height="451" alt="image" src="https://github.com/user-attachments/assets/5be8ae18-2979-4ea0-a283-7356f5de55b0" />

* the value above the shadow area means it is reliable / in 95% confidence area

# Partial Correlation Function 
<img width="1825" height="731" alt="image" src="https://github.com/user-attachments/assets/35dac0cd-d42b-4a36-844c-492e01d5cb8f" />

```
# 导入绘图所需的函数
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt

# 绘制 PACF 图
plt.figure(figsize=(8, 4))
plot_pacf(co2_series, lags=40, method='ywm')  # 'ywm' 是 Yule-Walker Method
plt.title('Partial Autocorrelation (PACF) of CO₂ Series')
plt.show()

```

<img width="726" height="475" alt="image" src="https://github.com/user-attachments/assets/578033c1-b876-4834-b062-5871540b9eb6" />
<img width="1548" height="1232" alt="image" src="https://github.com/user-attachments/assets/d82a5adc-e069-4f34-862f-0077c8333fd1" />

# ADF Test 
<img width="1738" height="833" alt="image" src="https://github.com/user-attachments/assets/aef31d6c-e014-40bf-b5c2-b595a106e0b5" /> 
<img width="1761" height="600" alt="image" src="https://github.com/user-attachments/assets/c6eaaf57-b22d-46b4-b4d2-a93381a0209c" />

```
adf_stat_diff, adf_pvalue_diff, _, _, _, _ = adfuller(co2_diff)
print(f"ADF p-value after differencing: {adf_pvalue_diff:.3f}")

```

<img width="1724" height="937" alt="image" src="https://github.com/user-attachments/assets/eb2874ee-c839-4e87-b0f4-fd035a905147" />


# Moving Average 
 <img width="1556" height="1133" alt="image" src="https://github.com/user-attachments/assets/5dff5cd5-2d73-42f6-8950-c1ff4bbb50b0" /> 
 <img width="1482" height="727" alt="image" src="https://github.com/user-attachments/assets/1c615c2b-9d23-40b8-8930-43cd742ece34" /> 

 
# Auto Regression Model 

<img width="1792" height="938" alt="image" src="https://github.com/user-attachments/assets/01dd983f-9437-41cd-b9ed-4822ced74f8c" />

<img width="1792" height="401" alt="image" src="https://github.com/user-attachments/assets/074bac80-ed75-4df2-b3c5-504f857dec42" />

# Arima Model 

<img width="2751" height="1599" alt="image" src="https://github.com/user-attachments/assets/25db2dc5-a334-4d23-a40b-40ddcbd3d68d" />
<img width="1151" height="1268" alt="image" src="https://github.com/user-attachments/assets/50732c6b-8bdb-48fe-abe1-7f8757dfe658" />
<img width="1611" height="1203" alt="image" src="https://github.com/user-attachments/assets/9091d3a5-a38a-4fe1-bc15-ae38a5ab3a4b" />
<img width="1863" height="570" alt="image" src="https://github.com/user-attachments/assets/beea9ca7-32ba-431b-8146-3a2e25b81c56" />
<img width="1032" height="727" alt="image" src="https://github.com/user-attachments/assets/efaf163c-af98-4dde-ada3-406c03aef5e3" />
<img width="1624" height="453" alt="image" src="https://github.com/user-attachments/assets/a6f1749b-c83c-4f24-9759-7385a71f100c" />

