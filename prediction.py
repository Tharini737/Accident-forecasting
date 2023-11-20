import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose


#data cleaning 
raw_data = pd.read_csv("raw_data.csv", usecols= ['MONATSZAHL', 'AUSPRAEGUNG', 'JAHR', 'MONAT', 'WERT'])
def data_prep(*args):
    df_data = raw_data[raw_data['JAHR'] <= 2020]
    df_filtered = df_data.copy()
    df_filtered['MONAT'] = pd.to_datetime(df_filtered['MONAT'],format = '%Y%m')
    df_filtered.rename(columns = {'MONAT':'Time'},inplace = True)
    df_filtered.set_index(df_filtered['Time'],inplace = True)
    df_filtered.drop(columns = ['JAHR','Time'], inplace = True)
    df_filtered.sort_index(inplace = True)
    return df_filtered

#model definition
def predictions(category, accident_type, year, month):
    df_filtered = data_prep(raw_data)
    forecast_date = f"{year}-{str(month).zfill(2)}-01"

    category_combinations = df_filtered[['MONATSZAHL', 'AUSPRAEGUNG']].drop_duplicates()
    models = {}
    for _, row in category_combinations.iterrows():
        monatszahl = row['MONATSZAHL']
        auspraegung = row['AUSPRAEGUNG']

        # Filter the data for the current combination
        selected_series = df_filtered[(df_filtered['MONATSZAHL'] == monatszahl) & (df_filtered['AUSPRAEGUNG'] == auspraegung)]['WERT']

        # Check if the series is not empty
        if not selected_series.empty:
            model = SARIMAX(selected_series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12))
            results = model.fit()
            models[(monatszahl, auspraegung)] = results

    # Use the category and accident_type provided by the user to get the prediction
    if (category, accident_type) in models:
        forecast = models[(category, accident_type)].get_prediction(start=pd.to_datetime(forecast_date), end=pd.to_datetime(forecast_date))
        forecast_value = forecast.predicted_mean[0]
        return forecast_value
    else:
        return "Model for the specified category and accident type not found"
