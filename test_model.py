"""
================================================================================
Testing File for Sales Vision Project
================================================================================
Tests:
1. preprocess_data           - Data cleaning
2. aggregate_to_monthly      - Monthly aggregation
3. engineer_features         - Feature engineering
4. generate_business_insights - Business insights generation
5. API endpoints
================================================================================
How to run:
    pytest test_model.py -v
================================================================================
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================
# Small sample data used in all tests
# Instead of using the real 1.5 million rows
# ============================================================

def make_sample_data(n_days=400, include_nulls=False, include_negatives=False):
    """
    Creates small sample data for testing
    n_days: number of days
    """
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
    sales = np.random.uniform(500, 5000, n_days)

    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales,
        'Product_ID': np.random.choice(['P001', 'P002', 'P003'], n_days),
        'Product': np.random.choice(['Milk', 'Bread', 'Eggs'], n_days),
        'Category': np.random.choice(['Dairy', 'Bakery', 'Poultry'], n_days),
    })

    if include_nulls:
        # Insert null values in some rows
        df.loc[5:10, 'Sales'] = np.nan

    if include_negatives:
        # Insert negative values in some rows
        df.loc[0:3, 'Sales'] = -100

    return df


def make_monthly_data(n_months=36):
    """
    Creates ready monthly data for tests that need monthly input
    """
    dates = pd.date_range(start='2022-01-01', periods=n_months, freq='MS')
    sales = np.random.uniform(50000, 150000, n_months)
    return pd.DataFrame({'Date': dates, 'Sales': sales})


# ============================================================
# TEST 1: preprocess_data
# ============================================================

class TestPreprocessData:
    """Tests for the data preprocessing step"""

    def setup_method(self):
        """Runs before each test to create a fresh Pipeline"""
        from final_model import SARIMAXForecastingPipeline
        self.pipeline = SARIMAXForecastingPipeline()

    def test_output_is_dataframe(self):
        """Ensure the output is a DataFrame"""
        df = make_sample_data()
        result = self.pipeline.preprocess_data(df)
        assert isinstance(result, pd.DataFrame), "Output should be a DataFrame"

    def test_removes_null_sales(self):
        """Ensure null values in Sales column are removed"""
        df = make_sample_data(include_nulls=True)
        result = self.pipeline.preprocess_data(df)
        assert result['Sales'].isnull().sum() == 0, "There should be no null values in Sales"

    def test_removes_negative_sales(self):
        """Ensure negative sales values are removed"""
        df = make_sample_data(include_negatives=True)
        result = self.pipeline.preprocess_data(df)
        assert (result['Sales'] <= 0).sum() == 0, "There should be no negative or zero values in Sales"

    def test_date_column_is_datetime(self):
        """Ensure the Date column is in datetime format"""
        df = make_sample_data()
        result = self.pipeline.preprocess_data(df)
        assert pd.api.types.is_datetime64_any_dtype(result['Date']), "Date column should be datetime type"

    def test_raises_error_without_date_column(self):
        """Ensure a ValueError is raised if Date column is missing"""
        df = make_sample_data()
        df = df.drop(columns=['Date'])
        with pytest.raises(ValueError, match="Date"):
            self.pipeline.preprocess_data(df)

    def test_raises_error_without_sales_column(self):
        """Ensure a ValueError is raised if Sales column is missing"""
        df = make_sample_data()
        df = df.drop(columns=['Sales'])
        with pytest.raises(ValueError, match="Sales"):
            self.pipeline.preprocess_data(df)

    def test_result_is_sorted_by_date(self):
        """Ensure the output data is sorted by date ascending"""
        df = make_sample_data()
        # Reverse order to test if the function sorts correctly
        df = df.iloc[::-1].reset_index(drop=True)
        result = self.pipeline.preprocess_data(df)
        assert result['Date'].is_monotonic_increasing, "Data should be sorted from oldest to newest"


# ============================================================
# TEST 2: aggregate_to_monthly
# ============================================================

class TestAggregateToMonthly:
    """Tests for the monthly aggregation step"""

    def setup_method(self):
        from final_model import SARIMAXForecastingPipeline
        self.pipeline = SARIMAXForecastingPipeline()

    def test_output_has_sales_column(self):
        """Ensure the output contains a Sales column"""
        df = make_sample_data()
        clean = self.pipeline.preprocess_data(df)
        result = self.pipeline.aggregate_to_monthly(clean)
        assert 'Sales' in result.columns, "Output should contain a Sales column"

    def test_output_has_date_column(self):
        """Ensure the output contains a Date column"""
        df = make_sample_data()
        clean = self.pipeline.preprocess_data(df)
        result = self.pipeline.aggregate_to_monthly(clean)
        assert 'Date' in result.columns, "Output should contain a Date column"

    def test_monthly_sales_are_positive(self):
        """Ensure all monthly sales values are positive"""
        df = make_sample_data()
        clean = self.pipeline.preprocess_data(df)
        result = self.pipeline.aggregate_to_monthly(clean)
        assert (result['Sales'] > 0).all(), "All monthly sales should be positive"

    def test_number_of_months_is_correct(self):
        """Ensure the number of months is reasonable"""
        df = make_sample_data(n_days=400)
        clean = self.pipeline.preprocess_data(df)
        result = self.pipeline.aggregate_to_monthly(clean)
        # 400 days is approximately 13 months
        assert len(result) >= 12, "There should be at least 12 months in the output"

    def test_no_duplicate_months(self):
        """Ensure there are no duplicate months"""
        df = make_sample_data()
        clean = self.pipeline.preprocess_data(df)
        result = self.pipeline.aggregate_to_monthly(clean)
        assert result['Date'].nunique() == len(result), "There should be no duplicate months"


# ============================================================
# TEST 3: engineer_features
# ============================================================

class TestEngineerFeatures:
    """Tests for the feature engineering step"""

    def setup_method(self):
        from final_model import SARIMAXForecastingPipeline
        self.pipeline = SARIMAXForecastingPipeline()

    def test_creates_month_column(self):
        """Ensure the month column is created"""
        monthly = make_monthly_data(n_months=36)
        result = self.pipeline.engineer_features(monthly)
        assert 'month' in result.columns, "Output should contain a month column"

    def test_creates_lag_features(self):
        """Ensure lag features are created"""
        monthly = make_monthly_data(n_months=36)
        result = self.pipeline.engineer_features(monthly)
        assert 'sales_lag_1' in result.columns, "Output should contain sales_lag_1 column"
        assert 'sales_lag_3' in result.columns, "Output should contain sales_lag_3 column"
        assert 'sales_lag_12' in result.columns, "Output should contain sales_lag_12 column"

    def test_creates_rolling_features(self):
        """Ensure rolling features are created"""
        monthly = make_monthly_data(n_months=36)
        result = self.pipeline.engineer_features(monthly)
        assert 'rolling_mean_3' in result.columns, "Output should contain rolling_mean_3 column"
        assert 'rolling_mean_12' in result.columns, "Output should contain rolling_mean_12 column"

    def test_no_nulls_in_output(self):
        """Ensure there are no null values in the output"""
        monthly = make_monthly_data(n_months=36)
        result = self.pipeline.engineer_features(monthly)
        assert result.isnull().sum().sum() == 0, "There should be no null values in the output"

    def test_month_sin_cos_range(self):
        """Ensure sin and cos values are between -1 and 1"""
        monthly = make_monthly_data(n_months=36)
        result = self.pipeline.engineer_features(monthly)
        assert result['month_sin'].between(-1, 1).all(), "month_sin values must be between -1 and 1"
        assert result['month_cos'].between(-1, 1).all(), "month_cos values must be between -1 and 1"

    def test_exog_features_list_not_empty(self):
        """Ensure the exogenous features list is not empty"""
        monthly = make_monthly_data(n_months=36)
        self.pipeline.engineer_features(monthly)
        assert len(self.pipeline.exog_features) > 0, "The exog features list should not be empty"


# ============================================================
# TEST 4: generate_business_insights
# ============================================================

class TestGenerateBusinessInsights:
    """Tests for the business insights generation step"""

    def setup_method(self):
        from final_model import SARIMAXForecastingPipeline
        self.pipeline = SARIMAXForecastingPipeline()

    def _set_fake_forecast(self, values):
        """Inserts fake forecast results into the Pipeline for testing"""
        dates = pd.date_range(start='2025-01-01', periods=len(values), freq='MS')
        months = [d.strftime('%Y-%m') for d in dates]
        self.pipeline.forecast_results = pd.DataFrame({
            'Month': months,
            'Date': dates,
            'Forecast': values
        })

    def test_returns_list(self):
        """Ensure the output is a list"""
        self._set_fake_forecast([100000, 120000, 90000])
        result = self.pipeline.generate_business_insights()
        assert isinstance(result, list), "Output should be a list"

    def test_returns_one_insight_per_month(self):
        """Ensure the number of insights equals the number of months"""
        self._set_fake_forecast([100000, 120000, 90000])
        result = self.pipeline.generate_business_insights()
        assert len(result) == 3, "There should be one insight per month"

    def test_insights_are_strings(self):
        """Ensure all insights are strings"""
        self._set_fake_forecast([100000, 120000, 90000])
        result = self.pipeline.generate_business_insights()
        for insight in result:
            assert isinstance(insight, str), "Each insight should be a string"

    def test_high_demand_insight(self):
        """Ensure a high-sales month generates an increase stock recommendation"""
        # Second month has very high sales
        self._set_fake_forecast([100000, 200000, 100000])
        result = self.pipeline.generate_business_insights()
        assert 'increasing' in result[1].lower() or 'increase' in result[1].lower(), \
            "High demand month should recommend increasing stock"

    def test_low_demand_insight(self):
        """Ensure a low-sales month generates a reduce stock recommendation"""
        self._set_fake_forecast([100000, 100000, 10000])
        result = self.pipeline.generate_business_insights()
        assert 'reducing' in result[2].lower() or 'reduce' in result[2].lower(), \
            "Low demand month should recommend reducing stock"

    def test_no_forecast_returns_message(self):
        """Ensure a proper message is returned when no forecast exists"""
        self.pipeline.forecast_results = None
        result = self.pipeline.generate_business_insights()
        assert len(result) == 1, "Should return exactly one message"
        assert isinstance(result[0], str), "The message should be a string"


# ============================================================
# TEST 5: API
# ============================================================

class TestAPI:
    """Tests for the FastAPI endpoints"""

    def setup_method(self):
        from fastapi.testclient import TestClient
        from final_model_api import app
        self.client = TestClient(app)

    def _make_csv_bytes(self, df):
        """Converts a DataFrame to CSV bytes to send to the API"""
        import io
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer.read()

    def test_predict_with_combined_file_returns_200(self):
        """Ensure the API works correctly with a combined file"""
        # SARIMAX needs enough data - sending 3 full years
        df = make_sample_data(n_days=1095)
        csv_bytes = self._make_csv_bytes(df)

        response = self.client.post(
            "/predict",
            files={"combined": ("data.csv", csv_bytes, "text/csv")},
            data={"forecast_steps": "3"}
        )
        assert response.status_code == 200, f"Expected 200 but got {response.status_code}"

    def test_predict_response_has_forecast_key(self):
        """Ensure the API response contains the forecast key"""
        df = make_sample_data(n_days=1095)
        csv_bytes = self._make_csv_bytes(df)

        response = self.client.post(
            "/predict",
            files={"combined": ("data.csv", csv_bytes, "text/csv")},
            data={"forecast_steps": "3"}
        )
        data = response.json()
        assert "forecast" in data, "Response should contain the forecast key"

    def test_predict_response_has_historical_key(self):
        """Ensure the API response contains the historical key"""
        df = make_sample_data(n_days=1095)
        csv_bytes = self._make_csv_bytes(df)

        response = self.client.post(
            "/predict",
            files={"combined": ("data.csv", csv_bytes, "text/csv")},
            data={"forecast_steps": "3"}
        )
        data = response.json()
        assert "historical" in data, "Response should contain the historical key"

    def test_predict_without_files_returns_400(self):
        """Ensure the API returns 400 when no files are uploaded"""
        response = self.client.post(
            "/predict",
            data={"forecast_steps": "3"}
        )
        assert response.status_code in [400, 422], \
            f"Expected 400 or 422 when no files provided, got {response.status_code}"

    def test_forecast_steps_affects_output_length(self):
        """Ensure the forecast length matches the forecast_steps parameter"""
        df = make_sample_data(n_days=1095)
        csv_bytes = self._make_csv_bytes(df)

        response = self.client.post(
            "/predict",
            files={"combined": ("data.csv", csv_bytes, "text/csv")},
            data={"forecast_steps": "6"}
        )
        data = response.json()
        assert "forecast" in data, "Response should contain the forecast key"
        assert len(data["forecast"]) == 6, "Forecast should contain exactly 6 months"


# ============================================================
# TEST 6: Model Forecast Accuracy (MAE and RMSE)
# ============================================================

class TestModelAccuracy:
    """Tests to validate that forecast accuracy is within acceptable range"""

    def setup_method(self):
        from final_model import SARIMAXForecastingPipeline
        self.pipeline = SARIMAXForecastingPipeline()

    def _train_pipeline(self, n_days=1095):
        """Helper: runs full pipeline and returns forecast"""
        df = make_sample_data(n_days=n_days)
        clean = self.pipeline.preprocess_data(df)
        monthly = self.pipeline.aggregate_to_monthly(clean)
        features = self.pipeline.engineer_features(monthly)
        self.pipeline.train_sarimax(features, validate=False)
        forecast = self.pipeline.forecast_future(features, steps=3)
        return forecast, monthly

    def test_forecast_values_are_positive(self):
        """Ensure all forecasted values are positive numbers"""
        forecast, _ = self._train_pipeline()
        assert (forecast['Forecast'] > 0).all(), "All forecast values should be positive"

    def test_forecast_values_are_not_null(self):
        """Ensure no null values exist in the forecast"""
        forecast, _ = self._train_pipeline()
        assert forecast['Forecast'].isnull().sum() == 0, "Forecast should have no null values"

    def test_forecast_has_confidence_interval(self):
        """Ensure the forecast contains upper and lower confidence interval columns"""
        forecast, _ = self._train_pipeline()
        assert 'Upper_CI_95' in forecast.columns, "Forecast should contain Upper_CI_95 column"
        assert 'Lower_CI_95' in forecast.columns, "Forecast should contain Lower_CI_95 column"

    def test_upper_ci_greater_than_lower_ci(self):
        """Ensure upper confidence interval is always greater than lower"""
        forecast, _ = self._train_pipeline()
        assert (forecast['Upper_CI_95'] >= forecast['Lower_CI_95']).all(), \
            "Upper CI should always be greater than or equal to Lower CI"

    def test_forecast_mae_is_reasonable(self):
        """Ensure MAE is within a reasonable range compared to average sales"""
        forecast, monthly = self._train_pipeline()
        avg_sales = monthly['Sales'].mean()
        avg_forecast = forecast['Forecast'].mean()
        mae = abs(avg_sales - avg_forecast)
        # MAE should be less than 80% of average sales (reasonable threshold)
        assert mae < avg_sales * 0.8, \
            f"MAE {mae:.0f} is too high compared to average sales {avg_sales:.0f}"

    def test_forecast_not_constant(self):
        """Ensure the forecast is not returning the same value for all months"""
        forecast, _ = self._train_pipeline()
        assert forecast['Forecast'].nunique() > 1, \
            "Forecast should have different values per month, not all the same"


# ============================================================
# TEST 7: Cross Selling Recommendations
# ============================================================

class TestCrossSelling:
    """Tests for cross selling recommendations"""

    def setup_method(self):
        from final_model import SARIMAXForecastingPipeline
        self.pipeline = SARIMAXForecastingPipeline()

    def _make_cross_selling_data(self, n_days=365):
        """Creates data suitable for cross selling analysis"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Sales': np.random.uniform(500, 5000, n_days),
            'Product_ID': np.random.choice(['P001', 'P002', 'P003', 'P004'], n_days),
            'Product': np.random.choice(['Milk', 'Bread', 'Eggs', 'Butter'], n_days),
            'Category': np.random.choice(['Dairy', 'Bakery'], n_days),
        })
        return df

    def test_cross_selling_returns_list(self):
        """Ensure cross selling returns a list"""
        df = self._make_cross_selling_data()
        clean = self.pipeline.preprocess_data(df)
        result = self.pipeline.get_cross_selling_recommendations(clean_data=clean, min_support=0.01)
        assert isinstance(result, list), "Cross selling output should be a list"

    def test_cross_selling_items_have_required_keys(self):
        """Ensure each recommendation has antecedent, consequent, and lift keys"""
        df = self._make_cross_selling_data()
        clean = self.pipeline.preprocess_data(df)
        result = self.pipeline.get_cross_selling_recommendations(clean_data=clean, min_support=0.01)
        if len(result) > 0:
            for item in result:
                assert 'antecedent' in item, "Each recommendation should have antecedent key"
                assert 'consequent' in item, "Each recommendation should have consequent key"
                assert 'lift' in item, "Each recommendation should have lift key"

    def test_cross_selling_lift_is_positive(self):
        """Ensure all lift values are positive"""
        df = self._make_cross_selling_data()
        clean = self.pipeline.preprocess_data(df)
        result = self.pipeline.get_cross_selling_recommendations(clean_data=clean, min_support=0.01)
        if len(result) > 0:
            for item in result:
                assert item['lift'] > 0, "Lift value should always be positive"

    def test_cross_selling_no_self_recommendation(self):
        """Ensure a product is never recommended with itself"""
        df = self._make_cross_selling_data()
        clean = self.pipeline.preprocess_data(df)
        result = self.pipeline.get_cross_selling_recommendations(clean_data=clean, min_support=0.01)
        for item in result:
            assert item['antecedent'] != item['consequent'], \
                "A product should never be recommended with itself"

    def test_cross_selling_max_15_recommendations(self):
        """Ensure the number of recommendations does not exceed 15"""
        df = self._make_cross_selling_data()
        clean = self.pipeline.preprocess_data(df)
        result = self.pipeline.get_cross_selling_recommendations(clean_data=clean, min_support=0.01)
        assert len(result) <= 15, "Should return at most 15 recommendations"

    def test_cross_selling_handles_missing_product_column(self):
        """Ensure it returns empty list if no product column is found"""
        df = make_sample_data()
        # Remove all possible product columns
        df = df.drop(columns=['Product', 'Product_ID'])
        clean = self.pipeline.preprocess_data(df)
        result = self.pipeline.get_cross_selling_recommendations(clean_data=clean)
        assert result == [], "Should return empty list when no product column exists"


# ============================================================
# TEST 8: API Edge Cases (Corrupted and Empty Files)
# ============================================================

class TestAPIEdgeCases:
    """Tests for API behavior with bad or corrupted input files"""

    def setup_method(self):
        from fastapi.testclient import TestClient
        from final_model_api import app
        self.client = TestClient(app)

    def test_corrupted_file_returns_error(self):
        """Ensure the API returns an error when the file content is corrupted"""
        corrupted_bytes = b"this is not a valid csv !!! @@@###"
        response = self.client.post(
            "/predict",
            files={"combined": ("data.csv", corrupted_bytes, "text/csv")},
            data={"forecast_steps": "3"}
        )
        # Should return an error, not crash silently
        assert response.status_code in [400, 422, 500], \
            "API should return an error for corrupted files"

    def test_empty_file_returns_error(self):
        """Ensure the API returns an error when the file is completely empty"""
        empty_bytes = b""
        response = self.client.post(
            "/predict",
            files={"combined": ("data.csv", empty_bytes, "text/csv")},
            data={"forecast_steps": "3"}
        )
        assert response.status_code in [400, 422, 500], \
            "API should return an error for empty files"

    def test_file_missing_sales_column_returns_error(self):
        """Ensure the API returns an error when Sales column is missing"""
        import io
        df = make_sample_data()
        df = df.drop(columns=['Sales'])
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        response = self.client.post(
            "/predict",
            files={"combined": ("data.csv", buffer.read(), "text/csv")},
            data={"forecast_steps": "3"}
        )
        assert response.status_code in [400, 422, 500], \
            "API should return an error when Sales column is missing"

    def test_file_missing_date_column_returns_error(self):
        """Ensure the API returns an error when Date column is missing"""
        import io
        df = make_sample_data()
        df = df.drop(columns=['Date'])
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        response = self.client.post(
            "/predict",
            files={"combined": ("data.csv", buffer.read(), "text/csv")},
            data={"forecast_steps": "3"}
        )
        assert response.status_code in [400, 422, 500], \
            "API should return an error when Date column is missing"

    def test_only_sales_file_without_products_returns_400(self):
        """Ensure the API returns error when only sales file is sent without products and calendar"""
        import io
        df = make_sample_data()
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        response = self.client.post(
            "/predict",
            files={"sales": ("sales.csv", buffer.read(), "text/csv")},
            data={"forecast_steps": "3"}
        )
        assert response.status_code in [400, 422], \
            "API should return 400 when only one of three files is sent"


# ============================================================
# TEST 9: Concurrent Requests
# ============================================================

class TestConcurrentRequests:
    """Tests for API behavior when multiple requests are sent at the same time"""

    def setup_method(self):
        from fastapi.testclient import TestClient
        from final_model_api import app
        self.client = TestClient(app)

    def _make_csv_bytes(self, df):
        """Converts a DataFrame to CSV bytes"""
        import io
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer.read()

    def test_two_sequential_requests_both_succeed(self):
        """Ensure two requests sent one after the other both return 200"""
        df = make_sample_data(n_days=1095)
        csv_bytes = self._make_csv_bytes(df)

        response1 = self.client.post(
            "/predict",
            files={"combined": ("data.csv", csv_bytes, "text/csv")},
            data={"forecast_steps": "3"}
        )
        response2 = self.client.post(
            "/predict",
            files={"combined": ("data.csv", csv_bytes, "text/csv")},
            data={"forecast_steps": "3"}
        )
        assert response1.status_code == 200, f"First request failed with {response1.status_code}"
        assert response2.status_code == 200, f"Second request failed with {response2.status_code}"

    def test_two_requests_with_different_steps_return_correct_lengths(self):
        """Ensure two requests with different forecast_steps each return the correct length"""
        df = make_sample_data(n_days=1095)
        csv_bytes = self._make_csv_bytes(df)

        response3 = self.client.post(
            "/predict",
            files={"combined": ("data.csv", csv_bytes, "text/csv")},
            data={"forecast_steps": "3"}
        )
        response6 = self.client.post(
            "/predict",
            files={"combined": ("data.csv", csv_bytes, "text/csv")},
            data={"forecast_steps": "6"}
        )

        data3 = response3.json()
        data6 = response6.json()

        assert len(data3["forecast"]) == 3, "First request should return 3 months forecast"
        assert len(data6["forecast"]) == 6, "Second request should return 6 months forecast"

    def test_valid_and_invalid_request_together(self):
        """Ensure a valid request succeeds even after an invalid one"""
        # First: send invalid request
        self.client.post(
            "/predict",
            data={"forecast_steps": "3"}
        )

        # Second: send valid request - should still work fine
        df = make_sample_data(n_days=1095)
        csv_bytes = self._make_csv_bytes(df)
        response = self.client.post(
            "/predict",
            files={"combined": ("data.csv", csv_bytes, "text/csv")},
            data={"forecast_steps": "3"}
        )
        assert response.status_code == 200, \
            "Valid request should succeed even after an invalid one"
        