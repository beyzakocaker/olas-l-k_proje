from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

# Selenium WebDriver ayarları
driver = webdriver.Chrome(ChromeDriverManager().install())

# Nvidia ve Tesla veri çekme
def get_stock_data(stock_symbol, max_rows=1000):
    driver.get(f'https://www.nasdaq.com/market-activity/stocks/{stock_symbol}/historical')
    time.sleep(5)  # Sayfanın yüklenmesi için bekleme

    data = []
    
    while len(data) < max_rows:
        # Verileri içeren tabloyu al
        table = driver.find_element(By.XPATH, '//*[@id="historicalDataTable"]')
        rows = table.find_elements(By.TAG_NAME, 'tr')
        
        for row in rows[1:]:
            cols = row.find_elements(By.TAG_NAME, 'td')
            if len(cols) > 4:
                date = cols[0].text
                close_price = cols[4].text
                data.append([date, close_price])
        
        try:
            next_button = driver.find_element(By.XPATH, '//button[@aria-label="Next"]')
            if next_button.is_enabled():
                next_button.click()
                time.sleep(5)
            else:
                break
        except Exception as e:
            print("Son sayfaya ulaşıldı veya bir hata oluştu:", e)
            break

    data = data[:max_rows]
    
    df = pd.DataFrame(data, columns=["Date", "Close"])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Close'] = pd.to_numeric(df['Close'].str.replace(',', '').str.replace('$', ''), errors='coerce')
    return df

# Nvidia ve Tesla verilerini çekme
nvda_data = get_stock_data("nvda", max_rows=1000)
tsla_data = get_stock_data("tsla", max_rows=1000)

# Veriyi kaydetme
nvda_data.to_csv('nvda_stock_data.csv', index=False)
tsla_data.to_csv('tsla_stock_data.csv')

driver.quit()

# Veri yükleme ve tanımlayıcı istatistikler
nvda_df = pd.read_csv('nvda_stock_data.csv')
tsla_df = pd.read_csv('tsla_stock_data.csv')

print("Nvidia Hisse İstatistikleri")
print(nvda_df.describe())
print("\nTesla Hisse İstatistikleri")
print(tsla_df.describe())

# Görselleştirme
plt.figure(figsize=(12, 6))
sns.histplot(nvda_df['Close'], kde=True, color='blue', label='Nvidia')
sns.histplot(tsla_df['Close'], kde=True, color='red', label='Tesla')
plt.legend()
plt.title('Hisse Senedi Fiyatlarının Dağılımı')
plt.xlabel('Fiyat (USD)')
plt.ylabel('Frekans')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=[nvda_df['Close'], tsla_df['Close']], labels=['Nvidia', 'Tesla'])
plt.title('Hisse Senedi Fiyatları Kutu Grafiği')
plt.show()

# İstatistiksel Test ve Korelasyon Analizi
t_stat, p_value = ttest_ind(nvda_df['Close'].dropna(), tsla_df['Close'].dropna())
print(f"T-istatistiği: {t_stat}")
print(f"p-değeri: {p_value}")
if p_value < 0.05:
    print("Farklılık anlamlıdır!")
else:
    print("Farklılık anlamlı değildir.")

correlation = nvda_df['Close'].corr(tsla_df['Close'])
print(f"Nvidia ve Tesla hisseleri arasındaki korelasyon: {correlation}")

# Regresyon Analizi
X = nvda_df['Close'].dropna().values.reshape(-1, 1)
y = tsla_df['Close'].dropna().values
model = LinearRegression()
model.fit(X, y)

plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', label='Veri Noktaları')
plt.plot(X, model.predict(X), color='red', label='Regresyon Doğrusu')
plt.title('Nvidia vs Tesla: Regresyon Analizi')
plt.xlabel('Nvidia Fiyatı (USD)')
plt.ylabel('Tesla Fiyatı (USD)')
plt.legend()
plt.show()

print(f"Model Katsayısı: {model.coef_}")
print(f"Model Kesişimi: {model.intercept_}")

# Veri Normalizasyonu ve RNN için Hazırlık
scaler = MinMaxScaler(feature_range=(0, 1))

def prepare_data(df, time_step=60):
    data = df[['Close']].values
    data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

X_nvda, y_nvda = prepare_data(nvda_df)
X_tsla, y_tsla = prepare_data(tsla_df)

X_train_nvda, X_test_nvda = X_nvda[:int(0.8*len(X_nvda))], X_nvda[int(0.8*len(X_nvda)):]
y_train_nvda, y_test_nvda = y_nvda[:int(0.8*len(y_nvda))], y_nvda[int(0.8*len(y_nvda)):]

X_train_tsla, X_test_tsla = X_tsla[:int(0.8*len(X_tsla))], X_tsla[int(0.8*len(X_tsla)):]
y_train_tsla, y_test_tsla = y_tsla[:int(0.8*len(y_tsla))], y_tsla[int(0.8*len(y_tsla)):]

# RNN Modeli
def build_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

nvda_model = build_rnn_model((X_train_nvda.shape[1], 1))
tsla_model = build_rnn_model((X_train_tsla.shape[1], 1))

nvda_model.fit(X_train_nvda, y_train_nvda, epochs=10, batch_size=32)
tsla_model.fit(X_train_tsla, y_train_tsla, epochs=10, batch_size=32)

# Tahmin ve Görselleştirme
predicted_nvda = nvda_model.predict(X_test_nvda)
predicted_tsla = tsla_model.predict(X_test_tsla)

predicted_nvda = scaler.inverse_transform(predicted_nvda)
y_test_nvda = scaler.inverse_transform(y_test_nvda.reshape(-1, 1))

predicted_tsla = scaler.inverse_transform(predicted_tsla)
y_test_tsla = scaler.inverse_transform(y_test_tsla.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(y_test_nvda, color='blue', label='Gerçek Nvidia Fiyatı')
plt.plot(predicted_nvda, color='red', label='Tahmin Edilen Nvidia Fiyatı')
plt.title('Nvidia Hisse Fiyatı Tahmini')
plt.xlabel('Gün')
plt.ylabel('Fiyat (USD)')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test_tsla, color='blue', label='Gerçek Tesla Fiyatı')
plt.plot(predicted_tsla, color='red', label='Tahmin Edilen Tesla Fiyatı')
plt.title('Tesla Hisse Fiyatı Tahmini')
plt.xlabel('Gün')
plt.ylabel('Fiyat (USD)')
plt.legend()
plt.show()

