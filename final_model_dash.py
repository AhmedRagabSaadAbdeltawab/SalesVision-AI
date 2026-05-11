import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# --- إعدادات الصفحة ---
st.set_page_config(page_title="Pro-Sales AI Dashboard", layout="wide")

# استايل CSS موحد وشيك
st.markdown("""
    <style>
    .stMetric { background-color: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(128, 128, 128, 0.2); }
    .insight-card { border-left: 5px solid #10b981; background: rgba(16, 185, 129, 0.05); padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Pro-Sales AI: The Ultimate Forecasting Suite")

# --- 1. منطقة الرفع (في الصدارة) ---
with st.container():
    st.subheader("📂 ملفات البيانات والإعدادات")
    u_col1, u_col2, u_col3 = st.columns([1, 2, 1])
    
    with u_col1:
        upload_type = st.radio("طريقة الرفع", ("Combined File", "Separate Files"))
        forecast_steps = st.slider("Forecast Months", 1, 24, 12)
        
    with u_col2:
        files = {}
        if upload_type == "Combined File":
            combined_file = st.file_uploader("ارفع الملف المجمع", type=["csv"])
            if combined_file: files = {'combined': (combined_file.name, combined_file.getvalue(), 'text/csv')}
        else:
            sc1, sc2, sc3 = st.columns(3)
            with sc1: s_f = st.file_uploader("Sales", type=["csv"])
            with sc2: p_f = st.file_uploader("Products", type=["csv"])
            with sc3: c_f = st.file_uploader("Calendar", type=["csv"])
            if s_f and p_f and c_f:
                files = {'sales': (s_f.name, s_f.getvalue(), 'text/csv'), 
                         'products': (p_f.name, p_f.getvalue(), 'text/csv'), 
                         'calendar': (c_f.name, c_f.getvalue(), 'text/csv')}
    
    with u_col3:
        st.write("") # مسافة
        st.write("")
        run_button = st.button("Run Analysis", use_container_width=True)

# --- 2. معالجة النتائج ---
if run_button and files:
    with st.spinner("جاري التحليل الشامل..."):
        try:
            api_url = "http://127.0.0.1:7860/predict"
            response = requests.post(api_url, files=files, data={'forecast_steps': forecast_steps})
            
            if response.status_code == 200:
                res = response.json()
                df_forecast = pd.DataFrame(res['forecast'])
                df_hist = pd.DataFrame(res['historical'])
                df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
                df_hist['Date'] = pd.to_datetime(df_hist['Date'])
                residuals = res.get('residual', [])

                # إنشاء التبويبات (Tabs)
                tabs = st.tabs(["📈 التوقعات", "🔍 تحليل الأنماط", "💡 التنبيهات والروابط", "🛠 الجودة والبيانات"])

                # --- Tab 1: Forecasting (Line + Bar + Metrics) ---
                with tabs[0]:
                    # عرض الـ Metrics القديمة
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Forecasted Sales", f"{df_forecast['Forecast'].sum():,.0f} L.E")
                    m2.metric("Peak Month", f"{df_forecast['Forecast'].max():,.0f} L.E")
                    m3.metric("Growth Rate", f"{(df_forecast['Forecast'].pct_change().mean()*100):.1f}%")

                    # الرسم البياني الخطي القديم (Line Chart) مع الـ Confidence Interval
                    st.subheader("Historical vs Forecasted Trend")
                    fig_line = go.Figure()
                    fig_line.add_trace(go.Scatter(x=df_hist['Date'], y=df_hist['Sales'], name='Actual', line=dict(color='#2563eb')))
                    fig_line.add_trace(go.Scatter(x=df_forecast['Date'], y=df_forecast['Forecast'], name='AI Forecast', line=dict(color='#10b981', width=3)))
                    fig_line.add_trace(go.Scatter(
                        x=df_forecast['Date'].tolist() + df_forecast['Date'].tolist()[::-1],
                        y=df_forecast['Upper_CI_95'].tolist() + df_forecast['Lower_CI_95'].tolist()[::-1],
                        fill='toself', fillcolor='rgba(16, 185, 129, 0.1)', line=dict(color='rgba(0,0,0,0)'), name='Confidence'
                    ))
                    st.plotly_chart(fig_line, use_container_width=True)

                    # الرسم البياني بالأعمدة الجديد (Bar Chart)
                    st.subheader("Monthly Forecast Breakdown")
                    fig_bar = px.bar(df_forecast, x='Date', y='Forecast', text_auto='.2s', color_discrete_sequence=['#10b981'])
                    st.plotly_chart(fig_bar, use_container_width=True)

                # --- Tab 2: Seasonality & Trend ---
                with tabs[1]:
                    st.subheader("Decomposition of Sales Data")
                    df_decomp = df_hist.set_index('Date')
                    decomp = seasonal_decompose(df_decomp['Sales'], model='additive', period=12)
                    
                    st.write("**Trend Component (الاتجاه العام)**")
                    st.line_chart(decomp.trend)
                    
                    st.write("**Seasonal Component (النمط الموسمي المتكرر)**")
                    st.line_chart(decomp.seasonal)

                # --- Tab 3: Alerts & Cross-Selling ---
                with tabs[2]:
                    col_l, col_r = st.columns(2)
                    with col_l:
                        st.subheader("📢 Business Alerts")
                        for alert in res.get('inventory_alerts', []):
                            st.info(alert)
                    
                    with col_r:
                        st.subheader("🛒 Cross-Selling")
                        cs_data = res.get('cross_selling_recommendations', [])
                        if cs_data:
                            for item in cs_data[:5]:
                                st.markdown(f"""<div class='insight-card'>
                                <b>Buy:</b> {item['antecedent']} ➡️ <b>Suggest:</b> {item['consequent']}<br>
                                <small>Lift: {item['lift']}</small></div>""", unsafe_allow_html=True)

                # --- Tab 4: Quality & Raw Data ---
                with tabs[3]:
                    
                    st.subheader("Raw Prediction Table")
                    st.dataframe(df_forecast)

            else:
                st.error(f"API Error: {response.text}")
        except Exception as e:
            st.error(f"Error: {e}")