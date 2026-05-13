from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
from typing import Optional

from final_model import SARIMAXForecastingPipeline

app = FastAPI()

pipeline = SARIMAXForecastingPipeline(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
pipeline.model_fit = None


@app.post("/train")
async def train(
    combined: Optional[UploadFile] = File(None),
    sales: Optional[UploadFile] = File(None),
    products: Optional[UploadFile] = File(None),
    calendar: Optional[UploadFile] = File(None),
):
    try:
        data = None

        if combined:
            content = await combined.read()
            data = pd.read_csv(io.BytesIO(content))

        elif sales and products and calendar:
            sales_content = await sales.read()
            products_content = await products.read()
            calendar_content = await calendar.read()

            df_sales = pd.read_csv(io.BytesIO(sales_content))
            df_products = pd.read_csv(io.BytesIO(products_content))
            df_calendar = pd.read_csv(io.BytesIO(calendar_content))

            df_sales['Date'] = pd.to_datetime(df_sales['Date'])
            df_calendar['Date'] = pd.to_datetime(df_calendar['Date'])

            data = df_sales.merge(df_products, on='Product_ID', how='left')
            data = data.merge(df_calendar, on='Date', how='left')

        else:
            raise HTTPException(status_code=400, detail="Please upload 1 combined file or 3 separate files")

        clean_data = pipeline.preprocess_data(data)
        monthly_data = pipeline.aggregate_to_monthly(clean_data)
        feat_data = pipeline.engineer_features(monthly_data)
        pipeline.train_sarimax(feat_data, validate=False)

        return {"message": "Model trained successfully ✅"}

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict")
async def predict(forecast_steps: int = Form(12)):
    try:
        if pipeline.model_fit is None:
            raise HTTPException(status_code=400, detail="Model not trained yet. Call /train first.")

        forecast = pipeline.forecast_future(steps=forecast_steps)
        inventory_alerts = pipeline.generate_business_insights()
        cross_selling = pipeline.get_cross_selling_recommendations(min_support=0.02)

        result_package = {
            "forecast": forecast.to_dict(orient='records'),
            "inventory_alerts": inventory_alerts,
            "cross_selling_recommendations": cross_selling,
            "model_residuals": {
                "residuals": pipeline.model_fit.resid.tolist()[:100]
            }
        }
        return result_package

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7860)