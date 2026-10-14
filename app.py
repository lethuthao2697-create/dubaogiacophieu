import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
from datetime import datetime

# ==============================================================================
# CẤU HÌNH TRANG & CỐ ĐỊNH SEED (ỔN ĐỊNH KẾT QUẢ AI)
# ==============================================================================
st.set_page_config(page_title="Dự Đoán Giá Cổ Phiếu LSTM", layout="wide")

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Tự động chèn CSS cấu hình màu sắc giao diện tông Xanh Dương bắt mắt
st.markdown("""
    <style>
        /* Màu chữ tiêu đề chính */
        h1 { color: #0056b3 !important; font-weight: 700; }
        /* Màu chữ tiêu đề phụ */
        h2, h3 { color: #007bff !important; }
        /* Định dạng các khối thông báo mẹo nhỏ */
        .stAlert { border-left: 5px solid #007bff; background-color: #f0f7ff; }
        /* Tùy chỉnh màu đường kẻ ngang phân cách */
        hr { border-top: 2px solid #b3d7ff; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# GIAO DIỆN MENU BÊN TRÁI (SIDEBAR)
# ==============================================================================
st.sidebar.markdown("<h2 style='color: #0056b3; text-align: center;'>🔵 CẤU HÌNH</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

ticker = st.sidebar.text_input("🔹 Nhập mã cổ phiếu (yfinance format):", value="VNM.VN").upper()
start_date = st.sidebar.date_input("📅 Ngày bắt đầu phân tích:", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("📆 Ngày kết thúc phân tích:", datetime(2026, 10, 14))
forecast_days = st.sidebar.slider("🔮 Số ngày muốn dự đoán tương lai:", min_value=10, max_value=30, value=10)

st.sidebar.markdown("---")
run_analysis = st.sidebar.button("🚀 Bắt đầu Phân tích & Dự đoán", use_container_width=True)

# Giao diện chính giữa màn hình
st.title("📈 Ứng Dụng Dự Đoán Giá Đóng Cửa Cổ Phiếu Bằng LSTM")
st.markdown("<p style='font-size: 16px; color: #555;'>Hệ thống sử dụng mạng thần kinh nhân tạo học sâu hồi quy (LSTM) để phân tích xu hướng chuỗi thời gian.</p>", unsafe_allow_html=True)
st.markdown("---")

# ==============================================================================
# CÁC HÀM XỬ LÝ LOGIC (SỬ DỤNG CACHE ĐỂ TĂNG TỐC ĐỘ APP)
# ==============================================================================
@st.cache_data(show_spinner="⏳ Đang tải dữ liệu từ Yahoo Finance...")
def load_data(ticker_symbol, start, end):
    df_stock = yf.download(ticker_symbol, start=start, end=end)
    return df_stock

def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

@st.cache_resource(show_spinner="🧠 Đang huấn luyện mô hình học sâu LSTM (Có thể mất 1-2 phút)...")
def train_lstm_model(data_values):
    train_size = int(0.8 * len(data_values))
    train_data = data_values[:train_size]
    test_data = data_values[train_size:]
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    X_train, y_train = create_sequences(train_scaled)
    test_combined = np.vstack((train_scaled[-60:], test_scaled))
    X_test, y_test = create_sequences(test_combined)
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    val_size = int(0.1 * len(X_train))
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_final = X_train[:-val_size]
    y_train_final = y_train[:-val_size]
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        shuffle=False,   
        callbacks=[early_stop],
        verbose=0
    )
    
    test_pred = model.predict(X_test, verbose=0)
    test_pred = scaler.inverse_transform(test_pred)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1,1))
    
    return model, scaler, y_test_real, test_pred, train_scaled, test_scaled

# ==============================================================================
# ĐOẠN XỬ LÝ KHI NGƯỜI DÙNG BẤM NÚT CHẠY APP
# ==============================================================================
if run_analysis:
    if start_date >= end_date:
        st.error("❌ Lỗi: Ngày bắt đầu phải nhỏ hơn ngày kết thúc phân tích!")
    else:
        df = load_data(ticker, start_date, end_date)
        
        if df.empty:
            st.error(f"❌ Không tìm thấy dữ liệu cho mã cổ phiếu '{ticker}'. Bạn hãy kiểm tra lại ký tự xem đúng chưa nhé!")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"🔠 **Mã cổ phiếu:** {ticker}")
            with col2:
                st.info(f"📅 **Từ ngày:** {start_date}")
            with col3:
                st.info(f"📆 **Đến ngày:** {end_date}")
            
            st.subheader("📋 Bảng thông tin tổng quát về giá cổ phiếu (10 phiên mới nhất)")
            st.dataframe(df.tail(10).style.format("{:.2f}").background_gradient(cmap='Blues'), use_container_width=True)
            
            data_values = df[['Close']].values
            model, scaler, y_test_real, test_pred, train_scaled, test_scaled = train_lstm_model(data_values)
            
            st.markdown("---")
            st.subheader("🎯 Kết quả kiểm định mô hình (Đo lường trên tập Test)")
            rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
            mae = mean_absolute_error(y_test_real, test_pred)
            r2 = r2_score(y_test_real, test_pred)
            
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Chỉ số RMSE (Sai số bình phương)", f"{rmse:.2f}")
            m_col2.metric("Chỉ số MAE (Sai số tuyệt đối trung bình)", f"{mae:.2f}")
            m_col3.metric("Chỉ số R² Score (Độ chính xác)", f"{r2:.4f}")
            
            last_60_days = np.vstack((train_scaled, test_scaled))[-60:]
            last_60_days = last_60_days.reshape(1, 60, 1)
            
            future_predictions = []
            for i in range(forecast_days):
                next_pred = model.predict(last_60_days, verbose=0)
                future_predictions.append(next_pred)
                last_60_days = np.append(
                    last_60_days[:,1:,:],
                    next_pred.reshape(1,1,1),
                    axis=1
                )
            
            future_predictions = np.array(future_predictions).reshape(-1,1)
            future_predictions = scaler.inverse_transform(future_predictions)
            
            future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            df_forecast = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Giá dự đoán'])
            
            st.markdown("---")
            st.subheader(f"🔮 Kết quả dự đoán giá đóng cửa trong {forecast_days} ngày tới")
            
            f_col1, f_col2 = st.columns(2)
            
            with f_col1:
                st.write("**Bảng giá trị dự báo cụ thể:**")
                st.dataframe(df_forecast.style.format("{:.2f}").background_gradient(cmap='Blues'), use_container_width=True)
                
            with f_col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(y_test_real))), 
                    y=y_test_real.flatten(),
                    mode='lines',
                    name='Giá thật thực tế (Tập Test)',
                    line=dict(color='#002B49', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(len(test_pred))), 
                    y=test_pred.flatten(),
                    mode='lines',
                    name='Giá mô hình đoán thử',
                    line=dict(color='#64B5F6', width=2)
                ))
                future_x = list(range(len(y_test_real), len(y_test_real) + forecast_days))
                fig.add_trace(go.Scatter(
                    x=future_x, 
                    y=future_predictions.flatten(),
                    mode='lines+markers',
                    name=f'Dự báo {forecast_days} ngày tới',
                    line=dict(dash='dash', color='#007BFF', width=3),
                    marker=dict(size=6, color='#0056b3')
                ))
                fig.update_layout(
                    title=f"Biểu đồ phân tích và xu hướng giá cổ phiếu {ticker}",
                    xaxis_title="Số phiên thời gian giao dịch",
                    yaxis_title="Mức giá đóng cửa",
                    paper_bgcolor='#f8f9fa',
                    plot_bgcolor='#ffffff',
                    height=500
                )
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E3F2FD')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E3F2FD')
                
                st.plotly_chart(fig, use_container_width=True)
else:
