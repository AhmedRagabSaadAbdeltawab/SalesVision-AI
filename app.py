from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
from typing import Optional

# استيراد الـ Pipeline بتاعك
from final_model import SARIMAXForecastingPipeline

app = FastAPI()

# تعريف الـ Pipeline
pipeline = SARIMAXForecastingPipeline(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

@app.post("/predict")
async def predict(
    combined: Optional[UploadFile] = File(None),
    sales: Optional[UploadFile] = File(None),
    products: Optional[UploadFile] = File(None),
    calendar: Optional[UploadFile] = File(None),
    forecast_steps: int = Form(12)
):
    try:
        data = None

        # 1. معالجة الملف المجمع (Combined File)
        if combined:
            print("Processing combined file....")
            content = await combined.read()
            data = pd.read_csv(io.BytesIO(content))
        
        # 2. معالجة الـ 3 ملفات المنفصلة
        elif sales and products and calendar:
            print("Processing 3 separate files...")
            
            sales_content = await sales.read()
            products_content = await products.read()
            calendar_content = await calendar.read()

            df_sales = pd.read_csv(io.BytesIO(sales_content))
            df_products = pd.read_csv(io.BytesIO(products_content))
            df_calendar = pd.read_csv(io.BytesIO(calendar_content))

            # التحويل والدمج
            df_sales['Date'] = pd.to_datetime(df_sales['Date'])
            df_calendar['Date'] = pd.to_datetime(df_calendar['Date'])
            
            data = df_sales.merge(df_products, on='Product_ID', how='left')
            data = data.merge(df_calendar, on='Date', how='left')
        
        else:
            raise HTTPException(status_code=400, detail="Please upload 1 combined file or 3 separate files")

        # 3. تنفيذ خطوات الـ Pipeline
        if data is not None:
            clean_data = pipeline.preprocess_data(data)
            monthly_data = pipeline.aggregate_to_monthly(clean_data)
            feat_data = pipeline.engineer_features(monthly_data)
            
            # تدريب الـ model
            pipeline.train_sarimax(feat_data, validate=False)
            
            # حساب التوقعات
            forecast = pipeline.forecast_future(feat_data, steps=forecast_steps)
            
            # Inventory Insights
            inventory_alerts = pipeline.generate_business_insights()

            # استخراج علاقات من المنتجات وبعض (cross-selling)
            cross_selling = pipeline.get_cross_selling_recommendations(min_support=0.02,clean_data=clean_data) 
            result_package = {
                "forecast": forecast.to_dict(orient='records'),
                "historical": monthly_data.reset_index().to_dict(orient='records'),
                "inventory_alerts": inventory_alerts,
                "cross_selling_recommendations": cross_selling,
                "model_residuals": {
                    
                    "residuals": pipeline.model_fit.resid.tolist()[:100] # نبعت اول 100 بس عشان ال size
                }
            }
            return result_package

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == '__main__':
    import uvicorn
    # تشغيل السيرفر
    uvicorn.run(app, host='0.0.0.0', port=7860)