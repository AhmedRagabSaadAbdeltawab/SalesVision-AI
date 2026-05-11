"""
================================================================================
SARIMAX TIME SERIES FORECASTING PIPELINE
================================================================================
Production-ready pipeline for sales forecasting using SARIMAX models

Features:
- Handles 3 separate files OR 1 combined file
- Automatic data preprocessing and feature engineering
- SARIMAX model with exogenous variables
- Produces 3-month forecasts with confidence intervals
- Fully configurable and reusable

Author: Senior Data Science Team
Date: February 2026
================================================================================
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Optional, Tuple, Dict, Union
import pickle

# Statistical modeling
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')


class SARIMAXForecastingPipeline:
    """
    Complete SARIMAX-based forecasting pipeline
    
    Handles:
    - Data loading (3 files or 1 combined file)
    - Preprocessing and cleaning
    - Feature engineering
    - SARIMAX model training
    - Forecasting with confidence intervals
    """
    
    def __init__(self, 
                 forecast_horizon: int = 3,
                 seasonal_period: int = 12,
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)):
        """
        Initialize SARIMAX pipeline
        
        Parameters:
        -----------
        forecast_horizon : int
            Number of months to forecast ahead (default: 3)
        seasonal_period : int
            Seasonal period in months (default: 12 for yearly seasonality)
        order : tuple (p, d, q)
            Non-seasonal ARIMA order
            p: AR order, d: differencing, q: MA order
        seasonal_order : tuple (P, D, Q, s)
            Seasonal ARIMA order
            P: seasonal AR, D: seasonal diff, Q: seasonal MA, s: period
        """
        self.forecast_horizon = forecast_horizon
        self.seasonal_period = seasonal_period
        self.order = order
        self.seasonal_order = seasonal_order
        
        # Storage for pipeline components
        self.monthly_sales = None
        self.model = None
        self.model_fit = None
        self.exog_features = []
        self.forecast_results = None
        
        print("="*80)
        print("SARIMAX FORECASTING PIPELINE INITIALIZED")
        print("="*80)
        print(f"Forecast Horizon: {forecast_horizon} months")
        print(f"SARIMAX Order: {order}")
        print(f"Seasonal Order: {seasonal_order}")
        print("="*80)
    
    
    def load_data(self, 
                  sales_path: Optional[str] = None,
                  products_path: Optional[str] = None,
                  calendar_path: Optional[str] = None,
                  combined_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from either 3 separate files OR 1 combined file
        
        Parameters:
        -----------
        sales_path : str, optional
            Path to Sales.csv
        products_path : str, optional
            Path to Products.csv
        calendar_path : str, optional
            Path to Calendar.csv
        combined_path : str, optional
            Path to combined dataset
            
        Returns:
        --------
        pd.DataFrame
            Raw merged/loaded data
        """
        print("\n[STEP 1] DATA LOADING")
        print("-"*80)
        
        if combined_path:
            # Load combined file
            print(f"Loading combined dataset from: {combined_path}")
            data = pd.read_csv(combined_path)
            print(f"Loaded {len(data):,} records from combined file")
            
        elif all([sales_path, products_path, calendar_path]):
            # Load three separate files
            print("Loading from 3 separate files:")
            
            # Load each file
            df_sales = pd.read_csv(sales_path)
            df_products = pd.read_csv(products_path)
            df_calendar = pd.read_csv(calendar_path)
            
            print(f"Sales: {len(df_sales):,} records")
            print(f"Products: {len(df_products):,} records")
            print(f"Calendar: {len(df_calendar):,} records")
            
            # Convert dates
            df_sales['Date'] = pd.to_datetime(df_sales['Date'])
            df_calendar['Date'] = pd.to_datetime(df_calendar['Date'])
            
            # تحويل كل ال product id في ال sales ل string عشان ال merge مايعملش مشاكل
            df_sales['Product_ID'] = df_sales['Product_ID'].astype(str)
            df_products['Product_ID'] = df_products['Product_ID'].astype(str)
            
            # Merge datasets
            # Step 1: Sales + Products (on Product_ID)
            data = df_sales.merge(df_products, on='Product_ID', how='left')
            
            # Step 2: Add Calendar data (on Date)
            data = data.merge(df_calendar, on='Date', how='left')
            
            print(f"\n Merged into {len(data):,} total records")
            
        else:
            raise ValueError(
                "Must provide either:\n"
                "  - combined_path (for 1 file), OR\n"
                "  - sales_path + products_path + calendar_path (for 3 files)"
            )
        
        self.raw_data = data
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        
        return data
    
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data
        
        Steps:
        - Handle missing values
        - Remove duplicates
        - Convert data types
        - Handle outliers
        - Validate essential columns
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw merged data
            
        Returns:
        --------
        pd.DataFrame
            Cleaned data
        """
        print("\n[STEP 2] DATA PREPROCESSING")
        print("-"*80)
        
        df = data.copy()
        original_size = len(df)
        
        # ================================================================
        # 1. ENSURE DATE COLUMN IS DATETIME
        # ================================================================
        if 'Date' not in df.columns:
            raise ValueError("Dataset must contain 'Date' column")
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Remove rows with invalid dates
        invalid_dates = df['Date'].isna().sum()
        if invalid_dates > 0:
            print(f" Removing {invalid_dates} rows with invalid dates")
            df = df.dropna(subset=['Date'])
        
        # ================================================================
        # 2. VALIDATE SALES COLUMN
        # ================================================================
        if 'Sales' not in df.columns:
            raise ValueError("Dataset must contain 'Sales' column (target variable)")
        
        # Convert to numeric
        df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
        
        # Remove negative or zero sales (if any)
        invalid_sales = (df['Sales'].isna()) | (df['Sales'] <= 0)
        if invalid_sales.sum() > 0:
            print(f" Removing {invalid_sales.sum()} rows with invalid sales values")
            df = df[~invalid_sales]
        
        # ================================================================
        # 3. HANDLE MISSING VALUES
        # ================================================================
        missing_before = df.isnull().sum().sum()
        
        if missing_before > 0:
            print(f"\nHandling {missing_before} missing values...")
            
            # For categorical columns: fill with mode
            cat_cols = df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col].fillna(mode_val[0], inplace=True)
            
            # For numerical columns: fill with median
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if col != 'Sales' and df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
            
            print(f" Missing values handled")
        
        # ================================================================
        # 4. REMOVE DUPLICATES
        # ================================================================
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f" Removing {duplicates} duplicate rows")
            df = df.drop_duplicates()
        
        # ================================================================
        # 5. HANDLE OUTLIERS (IQR method)
        # ================================================================
        Q1 = df['Sales'].quantile(0.25)
        Q3 = df['Sales'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df['Sales'] < lower_bound) | (df['Sales'] > upper_bound)).sum()
        df['Sales'] = np.clip(df['Sales'], lower_bound, upper_bound)
        print(f" Adjusted {outliers} outliers using Capping (IQR range).")
        
        # ================================================================
        # 6. SORT BY DATE
        # ================================================================
        df = df.sort_values('Date').reset_index(drop=True)
        
        removed = original_size - len(df)
        print(f"\n Preprocessing complete")
        print(f"  Removed: {removed:,} rows ({removed/original_size*100:.2f}%)")
        print(f"  Final size: {len(df):,} rows")
        
        self.clean_data = df
        return df
    
    
    def aggregate_to_monthly(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily sales to monthly level
        
        Parameters:
        -----------
        data : pd.DataFrame
            Cleaned data
            
        Returns:
        --------
        pd.DataFrame
            Monthly aggregated data with features
        """
        print("\n[STEP 3] MONTHLY AGGREGATION")
        print("-"*80)
        
        # Create year-month for grouping
        data['YearMonth'] = data['Date'].dt.to_period('M')
        
        # Aggregate sales to monthly
        monthly = data.groupby('YearMonth').agg({
            'Sales': 'sum'
        }).reset_index()
        
        # Convert period back to datetime (first day of month)
        monthly['Date'] = monthly['YearMonth'].dt.to_timestamp()
        monthly = monthly.drop('YearMonth', axis=1)
        
        # Sort by date
        monthly = monthly.sort_values('Date').reset_index(drop=True)
        
        print(f" Aggregated to monthly level")
        print(f"  Total months: {len(monthly)}")
        print(f"  Date range: {monthly['Date'].min().date()} to {monthly['Date'].max().date()}")
        print(f"  Average monthly sales: L.E{monthly['Sales'].mean():,.2f}")
        
        self.monthly_sales = monthly
        return monthly
    
    
    def engineer_features(self, monthly_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for SARIMAX model
        
        Features created:
        - Time-based: month, quarter, year
        - Lag features: t-1, t-3, t-12
        - Rolling statistics: 3-month and 12-month averages
        - Seasonal indicators
        
        Parameters:
        -----------
        monthly_data : pd.DataFrame
            Monthly aggregated data
            
        Returns:
        --------
        pd.DataFrame
            Data with engineered features
        """
        print("\n[STEP 4] FEATURE ENGINEERING")
        print("-"*80)
        
        df = monthly_data.copy()
        
        # ================================================================
        # TIME-BASED FEATURES
        # ================================================================
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        df['year'] = df['Date'].dt.year
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # ================================================================
        # LAG FEATURES (shift to prevent data leakage)
        # ================================================================
        df['sales_lag_1'] = df['Sales'].shift(1)
        df['sales_lag_3'] = df['Sales'].shift(3)
        df['sales_lag_12'] = df['Sales'].shift(12)
        
        # ================================================================
        # ROLLING STATISTICS
        # ================================================================
        df['rolling_mean_3'] = df['Sales'].shift(1).rolling(window=3, min_periods=1).mean()
        df['rolling_mean_12'] = df['Sales'].shift(1).rolling(window=12, min_periods=1).mean()
        df['rolling_std_3'] = df['Sales'].shift(1).rolling(window=3, min_periods=1).std()
        
        # ================================================================
        # PERCENTAGE CHANGE
        # ================================================================
        df['pct_change_1'] = df['Sales'].pct_change(1)
        
        # Remove NaN rows created by lags
        df_clean = df.dropna().reset_index(drop=True)
        
        # Store exogenous feature names for SARIMAX
        self.exog_features = [
            'month_sin', 'month_cos',
            'sales_lag_1', 'sales_lag_3', 'sales_lag_12',
            'rolling_mean_3', 'rolling_mean_12', 'rolling_std_3',
            'pct_change_1'
        ]
        
        print(f" Created {len(self.exog_features)} exogenous features:")
        for feat in self.exog_features:
            print(f"  - {feat}")
        
        print(f"\n Final dataset: {len(df_clean)} months")
        
        self.monthly_sales = df_clean
        return df_clean
    
    
    def check_stationarity(self, series: pd.Series, title: str = "Series") -> Dict:
        """
        Perform Augmented Dickey-Fuller test for stationarity
        
        Parameters:
        -----------
        series : pd.Series
            Time series to test
        title : str
            Name of series for display
            
        Returns:
        --------
        dict
            Test results
        """
        print(f"\n[STATIONARITY TEST] {title}")
        print("-"*60)
        
        # Remove any NaN values
        series_clean = series.dropna()
        
        # Perform ADF test
        result = adfuller(series_clean)
        
        output = {
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
        
        print(f"ADF Statistic: {output['test_statistic']:.6f}")
        print(f"p-value: {output['p_value']:.6f}")
        print(f"Critical Values:")
        for key, value in output['critical_values'].items():
            print(f" {key}: {value:.4f}")
        
        if output['is_stationary']:
            print("Series is STATIONARY (p < 0.05)")
        else:
            print("Series is NON-STATIONARY (p >= 0.05)")
            print("Differencing recommended")
        
        return output
    
    
    def train_sarimax(self, 
                      monthly_data: pd.DataFrame,
                      validate: bool = True) -> SARIMAX:
        """
        Train SARIMAX model
        
        Parameters:
        -----------
        monthly_data : pd.DataFrame
            Monthly data with features
        validate : bool
            Whether to perform validation on last 3 months
            
        Returns:
        --------
        SARIMAX results object
        """
        print("\n[STEP 5] SARIMAX MODEL TRAINING")
        print("="*80)
        
        # Prepare endogenous (target) and exogenous variables
        endog = monthly_data['Sales']
        exog = monthly_data[self.exog_features]
        
        # Check stationarity
        self.check_stationarity(endog, "Sales (Target Variable)")
        
        # ================================================================
        # TRAIN-VALIDATION SPLIT (if validation enabled)
        # ================================================================
        if validate:
            val_size = 3
            train_endog = endog[:-val_size]
            train_exog = exog[:-val_size]
            val_endog = endog[-val_size:]
            val_exog = exog[-val_size:]
            
            print(f"\nTraining set: {len(train_endog)} months")
            print(f"Validation set: {val_size} months")
        else:
            train_endog = endog
            train_exog = exog
            print(f"\nUsing all {len(train_endog)} months for training")
        
        # ================================================================
        # FIT SARIMAX MODEL
        # ================================================================
        print(f"\nFitting SARIMAX{self.order}x{self.seasonal_order}...")
        
        model = SARIMAX(
            endog=train_endog,
            exog=train_exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # Fit the model
        self.model_fit = model.fit(disp=False, maxiter=200)
        
        print(" Model fitted successfully")
        print(f"\nModel Summary:")
        print(f"AIC: {self.model_fit.aic:.2f}")
        print(f"BIC: {self.model_fit.bic:.2f}")
        print(f"Log-Likelihood: {self.model_fit.llf:.2f}")
        
        # ================================================================
        # VALIDATION (if enabled)
        # ================================================================
        if validate:
            print("\n" + "="*80)
            print("VALIDATION PERFORMANCE")
            print("="*80)
            
            # Forecast for validation period
            forecast_val = self.model_fit.forecast(steps=val_size, exog=val_exog)
            
            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            
            mae = mean_absolute_error(val_endog, forecast_val)
            rmse = np.sqrt(mean_squared_error(val_endog, forecast_val))
            mape = np.mean(np.abs((val_endog - forecast_val) / val_endog)) * 100
            
            print(f"MAE:  L.E {mae:,.2f}")
            print(f"RMSE: L.E {rmse:,.2f}")
            print(f"MAPE: {mape:.2f}%")
            
            # Store validation results
            self.validation_metrics = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }
        
        # ================================================================
        # RETRAIN ON FULL DATA
        # ================================================================
        print("\n" + "="*80)
        print("RETRAINING ON FULL DATASET")
        print("="*80)
        
        final_model = SARIMAX(
            endog=endog,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.model_fit = final_model.fit(disp=False, maxiter=200)
        
        print("Final model trained on all data")
        print(f"AIC: {self.model_fit.aic:.2f}")
        print(f"BIC: {self.model_fit.bic:.2f}")
        
        return self.model_fit
    
    
    def forecast_future(self, 
                       monthly_data: pd.DataFrame,
                       steps: Optional[int] = None) -> pd.DataFrame:
        """
        Generate forecasts for future months
        
        Parameters:
        -----------
        monthly_data : pd.DataFrame
            Historical monthly data
        horizon : int, optional
            Forecast horizon (uses self.forecast_horizon if None)
            
        Returns:
        --------
        pd.DataFrame
            Forecast results with confidence intervals
        """
        if steps is None:
            steps = self.forecast_horizon
        
        print("\n[STEP 6] GENERATING FORECASTS")
        print("="*80)
        
        # ================================================================
        # CREATE FUTURE EXOGENOUS VARIABLES
        # ================================================================
        print(f"Creating exogenous features for next {steps} months...")
        
        last_date = monthly_data['Date'].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=steps,
            freq='MS'
        )
        
        future_exog = []
        
        for i, date in enumerate(future_dates, 1):
            row = {
                'Date': date,
                'month_sin': np.sin(2 * np.pi * date.month / 12),
                'month_cos': np.cos(2 * np.pi * date.month / 12),
            }
            
            # Use last known values for lags (conservative approach)
            if i == 1:
                row['sales_lag_1'] = monthly_data['Sales'].iloc[-1]
                row['sales_lag_3'] = monthly_data['Sales'].iloc[-3]
                row['sales_lag_12'] = monthly_data['Sales'].iloc[-12] if len(monthly_data) >= 12 else monthly_data['Sales'].iloc[0]
            else:
                # For subsequent months, use recent values
                row['sales_lag_1'] = monthly_data['Sales'].iloc[-i+1]
                row['sales_lag_3'] = monthly_data['Sales'].iloc[-i-2] if i <= len(monthly_data)-2 else monthly_data['Sales'].iloc[0]
                row['sales_lag_12'] = monthly_data['Sales'].iloc[-12] if len(monthly_data) >= 12 else monthly_data['Sales'].iloc[0]
            
            # Rolling statistics from last known data
            row['rolling_mean_3'] = monthly_data['Sales'].tail(3).mean()
            row['rolling_mean_12'] = monthly_data['Sales'].tail(12).mean()
            row['rolling_std_3'] = monthly_data['Sales'].tail(3).std()
            row['pct_change_1'] = monthly_data['Sales'].pct_change(1).iloc[-1]
            
            future_exog.append(row)
        
        future_exog_df = pd.DataFrame(future_exog)
        
        # ================================================================
        # GENERATE FORECAST
        # ================================================================
        print("Generating forecast with confidence intervals...")
        
        forecast_result = self.model_fit.get_forecast(
            steps=steps,
            exog=future_exog_df[self.exog_features]
        )
        
        # Extract forecast values and confidence intervals
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int(alpha=0.05)  # 95% CI
        
        # ================================================================
        # CREATE RESULTS DATAFRAME
        # ================================================================
        results_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': forecast_mean.values,
            'Lower_CI_95': forecast_ci.iloc[:, 0].values,
            'Upper_CI_95': forecast_ci.iloc[:, 1].values
        })
        
        results_df['Month'] = results_df['Date'].dt.strftime('%B %Y')
        
        # ================================================================
        # DISPLAY RESULTS
        # ================================================================
        print("\n" + "="*80)
        print(f"FORECAST RESULTS - Next {steps} Months")
        print("="*80)
        print(results_df[['Month', 'Forecast', 'Lower_CI_95', 'Upper_CI_95']].to_string(index=False))
        
        print(f"\nTotal Forecast: L.E {results_df['Forecast'].sum():,.2f}")
        print(f"Average Monthly: L.E {results_df['Forecast'].mean():,.2f}")
        
        self.forecast_results = results_df
        return results_df
    
    def get_cross_selling_recommendations(self, clean_data, min_support=0.02):# 2% لو ملقاش منتج موجود او مجموعه مجمود في الداتا اللي هيشوفها يبقى مش بيعمل علاقه مع حاجه
        # تحليل لسلة المشتريات باستخدام عينه من ال الداتا بطريقة ال corss selling
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            print("\n" + "="*80)
            print("[BONUS] GENERATING CROSS-SELLING STRATEGY (Market Basket Analysis)")
            print("="*80)
            
            #دور ع اسم المنتج ايا كان هو ايه 
            pontential_cols = [ 'Product','product','product_name','Product_Name','Product_ID', 'Product ID', 'item_id', 'StockCode','product_id', 'ProductID', 'ProductId', 'productid']
            item_col = next((col for col in pontential_cols if col in clean_data.columns), None)

            if not item_col:
                print(f"Error: Could not find product column. Available columns: {list(clean_data.columns)}")
                return []
               # هنحدد اخر سنه ياخد منها عينات عشان نكون ضامنين التحديثات الاخيره 
            last_year = clean_data['Date'].dt.year.max()
            last_year_data = clean_data[clean_data['Date'].dt.year == last_year]
            # ونستخدم في التجميع التاريخ مع المنتجات 
            sample_data = last_year_data.groupby(last_year_data['Date'].dt.month).apply(
                lambda x: x.sample(n=min(500, len(x)), random_state=42)
            ) .reset_index(drop=True)
            # هنا هنحول الداتا ل matrix (pivot table) 
                        # تحويل الارقام ل 0و 1 (موجوده او لأ ) 

            basket = (sample_data.groupby(['Date', item_col]).size().unstack(fill_value=0).astype(bool)) 
           
            # تشغيل Apriori
            frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True, low_memory=True)
            
            if frequent_itemsets.empty:
                print("No frequent itemsets found with the given support threshold.")
                return []
            # استخراج القواعد
            rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1) # 1 عشان نضمن ان في علاقه قويه بين المنتجين

            if rules.empty:
                print("No association rules found.")
                return []
            
            rules = rules.sort_values('lift', ascending=False)
            # عرض اهم 10 نصايح 
            top_rules = rules.sort_values('lift', ascending=False).head(15)
            recommendations = []
            seen_pairs = set()  # لتتبع الأزواج التي تم عرضها بالفعل
            for index, row in top_rules.iterrows():
                antecedents = list(row['antecedents'])[0]  # المنتج اللي بيشتريه الزبون
                consequents = list(row['consequents'])[0]  # المنتج اللي ممكن يشتريه بعد ما يشتري الاول
                pair = tuple(sorted([str(antecedents), str(consequents)]))
                if pair not in seen_pairs and antecedents != consequents:
                    seen_pairs.add(pair)
                    lift = row['lift']

                    print(f"Tip: The customer who buys [{antecedents}] is more interested in [{consequents}]")
                    print(f" (Lift: {lift:.2f} - indicates strong association)")
                    print("-"*80)
                    # بنجهز الداتا للداشبورد
                    recommendations.append({
                        'antecedent': str(antecedents),
                        'consequent': str(consequents),
                        'lift': round(float(lift), 2)
                    })
                if len(recommendations) >= 15:
                    break
            return recommendations
        except ImportError:
            print("mlxtend library is required for cross-selling analysis. Please install it using 'pip install mlxtend'.")
            return []
        except Exception as e:
            print(f"An error occurred during cross-selling analysis: {e}")
            return []
        


    def save_pipeline(self, filename: str = 'sarimax_pipeline.pkl'):
         """
         Save the trained pipeline
        
         Parameters:
         -----------
         filename : str
            Output filename
         """
         print("\n[STEP 8] SAVING PIPELINE")
         print("-"*80)

         pipeline_data = {
             'model_fit': self.model_fit,
             'monthly_sales': self.monthly_sales,
             'exog_features': self.exog_features,
             'forecast_results': self.forecast_results,
             'order': self.order,
             'seasonal_order': self.seasonal_order,
             'forecast_horizon': self.forecast_horizon,
             'validation_metrics': getattr(self, 'validation_metrics', None),
             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
         }
        
         with open(filename, 'wb') as f:
             pickle.dump(pipeline_data, f)
        
         print(f" Pipeline saved: {filename}")
        
         # Save forecast to CSV
         csv_filename = filename.replace('.pkl', '_forecast.csv')
         self.forecast_results.to_csv(csv_filename, index=False)
         print(f" Forecast saved: {csv_filename}")
    
    def generate_business_insights(self) -> list:
        # اقترحات للعميل بناء على ال forecast
        if self.forecast_results is None:
            return ["No forecast results available to generate insights."]

        avg_forecast = self.forecast_results['Forecast'].mean()
        insights = []

        print ("\n "+"="*80)
        print ("BUSINESS ACTION INSIGHTS & RECOMMENDATIONS")
        print ("="*80)

        for index , row in self.forecast_results.iterrows():
            month = row['Month']
            forecast = row['Forecast']

            if forecast > avg_forecast * 1.2:
                msg = f"{month}: Suggest increasing stock by 20% because we are experiencing high demand. "
            elif forecast < avg_forecast * 0.8:
                msg = f"{month}: Consider reducing stock by 20% due to expected lower demand. "
            else:
                msg = f"{month}: Maintain current stock levels as demand is expected to be stable. "
            insights.append(msg)
            print(msg)
        print("-"*80)
        return insights
    

    def run_full_pipeline(self,
                         sales_path: Optional[str] = None,
                         products_path: Optional[str] = None,
                         calendar_path: Optional[str] = None,
                         combined_path: Optional[str] = None,
                         validate: bool = True,
                         save_results: bool = True) -> pd.DataFrame:
        """
        Execute complete forecasting pipeline end-to-end
        
        Parameters:
        -----------
        sales_path : str, optional
        products_path : str, optional
        calendar_path : str, optional
        combined_path : str, optional
        validate : bool
            Perform validation on last 3 months
        save_results : bool
            Save pipeline and results
            
        Returns:
        --------
        pd.DataFrame
            Forecast results
        """
        print("\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*22 + "SARIMAX FORECASTING PIPELINE" + " "*28 + "║")
        print("╚" + "="*78 + "╝")
        

        
        # Step 1: Load data
        data = self.load_data(sales_path, products_path, calendar_path, combined_path)
        
        # Step 2: Preprocess
        clean_data = self.preprocess_data(data)
        
        # Step 3: Aggregate to monthly
        monthly_data = self.aggregate_to_monthly(clean_data)
        
        # Step 4: Feature engineering
        monthly_features = self.engineer_features(monthly_data)
        
        # Step 5: Train SARIMAX
        self.train_sarimax(monthly_features, validate=validate)
        
        # Step 6: Generate forecast
        forecast = self.forecast_future(
    monthly_data=monthly_features,
    steps=self.forecast_horizon
)
        # Step 7: Generate business insights
        self.generate_business_insights()

        #step 8: Cross-selling recommendations
        self.get_cross_selling_recommendations()

        # Step 9: Save pipeline and results
        if save_results:
            self.save_pipeline()

        return forecast

        