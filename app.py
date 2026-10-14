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
# CẤU HÌNH TRANG & FIX SEED ĐỂ ỔN ĐỊNH KẾT QUẢ
# ==============================================================================
st.set_page_config(page_title="Dự Đoán Giá Cổ Phiếu LSTM", layout="wide")

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# ==============================================================================
# THANH MENU ĐIỀU KHIỂN BÊN TRÁI (SIDEBAR)
# ==============================================================================
st.sidebar.header("Cấu Hình Ứng Dụng")

# Nhập mã cổ phiếu (Mặc định ban đầu là VNM.VN)
ticker = st.sidebar.text_input("Nhập mã cổ phiếu (yfinance format):", value="VNM.VN").upper()

# Chọn khoảng thời gian phân tích lịch sử
start_date = st.sidebar.date_input("Ngày bắt đầu phân tích:", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("Ngày kết thúc phân tích:", datetime(2026, 10, 14))

# Chọn số ngày muốn dự đoán tương lai (Từ 10 đến 30 ngày)
forecast_days = st.sidebar.slider("Số ngày muốn dự đoán tương lai:", min_value=10, max_value=30, value=10)

# Nút bấm để kích hoạt tính toán
run_analysis = st.sidebar.button("Bắt đầu Phân tích & Dự đoán")

# ==============================================================================
# GIAO DIỆN CHÍNH TRÊN MÀN HÌNH
# ==============================================================================
st.title("📊 Ứng Dụng Dự Đoán Giá Đóng Cửa Cổ Phiếu Bằng LSTM")
st.markdown("---")

# ==============================================================================
# CÁC HÀM XỬ LÝ LOGIC DỮ LIỆU & HUẤN LUYỆN AI (Sử dụng Cache tối ưu tốc độ)
# ==============================================================================
@st.cache_data(show_spinner="Đang tải dữ liệu từ Yahoo Finance...")
def load_data(ticker_symbol, start, end):
    df_stock = yf.download(ticker_symbol, start=start, end=end)
    return df_stock

def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

@st.cache_resource(show_spinner="Đang huấn luyện mô hình LSTM (Có thể mất 1-2 phút)...")
def train_lstm_model(data_values):
    # Chia tập dữ liệu Train / Test theo tỷ lệ 80% / 20%
    train_size = int(0.8 * len(data_values))
    train_data = data_values[:train_size]
    test_data = data_values[train_size:]
    
    # Chuẩn hóa dữ liệu về khoảng [0, 1]
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Tạo các chuỗi thời gian (window_size=60 phiên)
    X_train, y_train = create_sequences(train_scaled)
    test_combined = np.vstack((train_scaled[-60:], test_scaled))
    X_test, y_test = create_sequences(test_combined)
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Trích ra 10% từ tập Train làm tập Validation nhằm theo dõi loss
    val_size = int(0.1 * len(X_train))
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_final = X_train[:-val_size]
    y_train_final = y_train[:-val_size]
    
    # Xây dựng cấu trúc mạng LSTM giống hệt code mẫu gốc
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Cơ chế dừng sớm khi loss trên tập val không giảm nữa
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Tiến hành huấn luyện mô hình ẩn danh log nền
    model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        shuffle=False,   
        callbacks=[early_stop],
        verbose=0
    )
    
    # Thử nghiệm dự báo trên tập Test để làm cơ sở kiểm định chỉ số
    test_pred = model.predict(X_test, verbose=0)
    test_pred = scaler.inverse_transform(test_pred)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1,1))
    
    return model, scaler, y_test_real, test_pred, train_scaled, test_scaled

# ==============================================================================
# QUY TRÌNH CHẠY CHƯƠNG TRÌNH KHI NGƯỜI DÙNG NHẤN NÚT
# ==============================================================================
if run_analysis:
    if start_date >= end_date:
        st.error("Lỗi: Ngày bắt đầu phân tích phải nhỏ hơn ngày kết thúc!")
    else:
        # Tải dữ liệu lịch sử của mã cổ phiếu
        df = load_data(ticker, start_date, end_date)
        
        if df.empty:
            st.error(f"Không tìm thấy dữ liệu cho mã '{ticker}'. Hãy kiểm tra lại ký tự hoặc đuôi thị trường (ví dụ: .VN)!")
        else:
            # 1. Hiển thị thông tin tổng quan của phiên phân tích
            col1, col2, col3 = st.columns(3)
            col1.metric("Mã cổ phiếu phân tích", ticker)
            col2.metric("Ngày bắt đầu dữ liệu", str(start_date))
            col3.metric("Ngày kết thúc dữ liệu", str(end_date))
            
            # Hiển thị bảng dữ liệu 10 phiên gần đây nhất
            st.subheader("📋 Bảng thông tin tổng quát về giá cổ phiếu (10 phiên mới nhất)")
            st.dataframe(df.tail(10).style.format("{:.2f}"), use_container_width=True)
            
            # Trích xuất giá trị đóng cửa 'Close' để nạp vào AI
            data = df[['Close']].values
            
            # Gọi hàm huấn luyện và dự báo tập kiểm định
            model, scaler, y_test_real, test_pred, train_scaled, test_scaled = train_lstm_model(data)
            
            # 2. Hiển thị kết quả kiểm định mô hình độ chính xác (Metrics)
            st.markdown("---")
            st.subheader("🎯 Kết quả kiểm định mô hình (Đo lường trên tập dữ liệu kiểm tra Test)")
            
            rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
            mae = mean_absolute_error(y_test_real, test_pred)
            r2 = r2_score(y_test_real, test_pred)
            
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Chỉ số RMSE (Sai số căn bình phương)", f"{rmse:.2f}")
            m_col2.metric("Chỉ số MAE (Sai số tuyệt đối TB)", f"{mae:.2f}")
            m_col3.metric("Chỉ số R² Score (Độ phù hợp mô hình)", f"{r2:.4f}")
            
            # 3. Tiến hành dự đoán cuốn chiếu theo số ngày tương lai được chọn (10 - 30 ngày)
            last_60_days = np.vstack((train_scaled, test_scaled))[-60:]
            last_60_days = last_60_days.reshape(1, 60, 1)
            
            future_predictions = []
            for i in range(forecast_days):
                next_pred = model.predict(last_60_days, verbose=0)
                future_predictions.append(next_pred[0,0])
                last_60_days = np.append(
                    last_60_days[:,1:,:],
                    next_pred.reshape(1,1,1),
                    axis=1
                )
            
            future_predictions = np.array(future_predictions).reshape(-1,1)
            future_predictions = scaler.inverse_transform(future_predictions)
            
            # Tạo lập mốc ngày tương lai tương ứng để dựng bảng dữ liệu trực quan
            future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            df_forecast = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Giá dự đoán'])
            
            # Hiển thị kết quả dự đoán tương lai dạng bảng song song biểu đồ tương tác Plotly
            st.markdown("---")
            st.subheader(f"🔮 Kết quả dự đoán giá đóng cửa trong {forecast_days} ngày tới")
            
            f_col1, f_col2 = st.columns([1, 2]) # Chia tỷ lệ cột bảng (nhỏ) và biểu đồ (rộng)
            
            with f_col1:
                st.write("Bảng giá trị dự đoán chi tiết:")
                st.dataframe(df_forecast.style.format("{:.2f}"), use_container_width=True)
                
            with f_col2:
                # Thiết lập vẽ đồ thị tương tác cao bằng thư viện Plotly
                fig = go.Figure()
                
                # Thêm đường giá trị thực tế tập Test lịch sử
                fig.add_trace(go.Scatter(
                    x=list(range(len(y_test_real))), 
                    y=y_test_real.flatten(),
                    mode='lines',
                    name='Giá thật (Tập Test)'
                ))
                
                # Thêm đường giá trị mô hình đã dự báo trên tập Test để đối chiếu
                fig.add_trace(go.Scatter(
                    x=list(range(len(test_pred))), 
                    y=test_pred.flatten(),
                    mode='lines',
                    name='Giá dự báo thử (Tập Test)'
                ))
                
                # Thêm đường nối tiếp biểu diễn các ngày dự đoán tương lai phía trước
                future_x = list(range(len(y_test_real), len(y_test_real) + forecast_days))
                fig.add_trace(go.Scatter(
                    x=future_x, 
                    y=future_predictions.flatten(),
                    mode='lines+markers',
                    name=f'Dự báo tương lai ({forecast_days} ngày)',
                    line=dict(dash='dash', color='red', width=3)
                ))
                
                fig.update_layout(
