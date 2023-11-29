import tkinter
import customtkinter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import numpy as np
import pandas as pd

variable = "TESLA"
forcasttype = ""
weightforcast = ""
checktick_seasonal = "off"
select_model = "Basic Naïve"
def correlation_event():
    textbox.delete("0.0", "end")
    excel_file = pd.ExcelFile('Forecast New.xlsx')

    nasdaq_data = excel_file.parse('NASDAQ')
    tesla_data = excel_file.parse('TESLA')
    microsoft_data = excel_file.parse('MICROSOFT')
    sp500_data = excel_file.parse('S&P500')

    nasdaq_close_prices = nasdaq_data['Close']
    tesla_close_prices = tesla_data['Close']
    microsoft_close_prices = microsoft_data['Close']
    sp500_close_prices = sp500_data['Close']

    correlation_nasdaq_tesla = nasdaq_close_prices.corr(tesla_close_prices)
    correlation_nasdaq_microsoft = nasdaq_close_prices.corr(microsoft_close_prices)
    correlation_nasdaq_sp500 = nasdaq_close_prices.corr(sp500_close_prices)
    correlation_tesla_microsoft = tesla_close_prices.corr(microsoft_close_prices)
    correlation_tesla_sp500 = tesla_close_prices.corr(sp500_close_prices)
    correlation_microsoft_sp500 = microsoft_close_prices.corr(sp500_close_prices)

    textbox.insert(customtkinter.END, f'NDX & TSLA: %.4f\n' % correlation_nasdaq_tesla)
    textbox.insert(customtkinter.END, f'NDX & MSFT: %.4f\n' % correlation_nasdaq_microsoft)
    textbox.insert(customtkinter.END, f'NDX & SPX: %.4f\n' % correlation_nasdaq_sp500)
    textbox.insert(customtkinter.END, f'TSLA & MSFT: %.4f\n' % correlation_tesla_microsoft)
    textbox.insert(customtkinter.END, f'TSLA & SPX: %.4f\n' % correlation_tesla_sp500)
    textbox.insert(customtkinter.END, f'MSFT & SPX: %.4f\n' % correlation_microsoft_sp500)

def checkweight_event(choice):
    global weightforcast
    weightforcast = choice
    if weightforcast == "MA = 3" or "MA = 5" or "MA = 7":
        button_event()
    if weightforcast == "W = 0.1" or "W = 0.5" or "W = 0.9":
        button_event()


def combobox_callback_model(choice):
    global select_model
    global weightforcast
    select_model = choice
    if select_model == "Basic Naïve":
        new_values = ["None"]
        combobox3.configure(values=new_values)
        combobox3.set(new_values[0])
        weightforcast = "None"
    if select_model == "Moving Average":
        new_values = ["MA = 3","MA = 5","MA = 7"]
        combobox3.configure(values=new_values)
        combobox3.set(new_values[0])
        weightforcast = "MA = 3"
    if select_model == "Modified Naïve":
        new_values = ["W = 0.1","W = 0.5","W = 0.9"]
        combobox3.configure(values=new_values)
        combobox3.set(new_values[0])
        weightforcast = "W = 0.1"
    if select_model == "Decomposition":
        new_values = ["None"]
        combobox3.configure(values=new_values)
        combobox3.set(new_values[0])
        weightforcast = "None"
    if select_model == "Holt-Winter":
        new_values = ["None"]
        combobox3.configure(values=new_values)
        combobox3.set(new_values[0])
        weightforcast = "None"

def combobox_callback(choice):
    global variable
    variable = choice
    if forcasttype == "autocorrelation":
        autocorrelation_event()
    if forcasttype == "seasonal":
        seasonal_event()
    if forcasttype == "trend_event":
        trend_event()
    if forcasttype == "select_model":
        button_event()

def seasonal_event():
    textbox.delete("0.0", "end")
    textbox2.delete("0.0", "end")
    global forcasttype
    forcasttype = "seasonal"
    ax.clear()
    file_path = 'Forecast New.xlsx'
    df = pd.read_excel(file_path, sheet_name=variable)
    df.set_index('Time', inplace=True)
    df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()
    decomposition = seasonal_decompose(df['Close'], model='additive', period=12)
    seasonal = decomposition.seasonal
    ax.plot(seasonal, label='Seasonal', color='green')
    ax.legend()
    canvas.draw()

def trend_event():
    textbox.delete("0.0", "end")
    textbox2.delete("0.0", "end")
    global forcasttype
    forcasttype = "trend_event"
    ax.clear()
    file_path = 'Forecast New.xlsx'
    df = pd.read_excel(file_path, sheet_name=variable)
    df.set_index('Time', inplace=True)
    df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()
    decomposition = seasonal_decompose(df['Close'], model='additive', period=12)
    trending = decomposition.trend
    ax.plot(trending, label='Trend', color='green')
    ax.legend()
    canvas.draw()
    
def autocorrelation_event():
    global forcasttype
    global checktick_seasonal
    forcasttype = "autocorrelation"
#### Autocorrelation Raw Data
    if checktick_seasonal == "off":
        textbox.delete("0.0", "end")
        textbox2.delete("0.0", "end")
        ax.clear()
        df = pd.read_excel('Forecast New.xlsx', sheet_name=variable)
        autocorrelation = acf(df['Close'], nlags=36, fft=True)
        for lag in range(1, len(autocorrelation)):
            asterisk = ' *' if autocorrelation[lag] > 0.2581988897471611 else ''
            textbox.insert(customtkinter.END, f'Lag {lag} : {autocorrelation[lag]:.4f}{asterisk}\n')
            if autocorrelation[lag] > 0.2581988897471611:
                if lag < 10:
                    textbox.tag_add('color', f'{lag}.8', f'{lag}.15') 
                    textbox.tag_config('color', foreground='red')
                else:
                    textbox.tag_add('color', f'{lag}.8', f'{lag}.15') 
                    textbox.tag_config('color', foreground='red')

        for lag, value in enumerate(autocorrelation[1:], start=1):
            color = '#fdb9c8' if value > 0.2581988897471611 else '#aec6cf'
            ax.bar(lag, value, color=color)

        ax.set_title('ACF')
        ax.set_xlabel('Lag Number')
        ax.set_ylabel('Autocorrelation')
        canvas.draw()
#### Autocorrelation D-Seasonal and D-Trend
    if checktick_seasonal == "on":
        textbox.delete("0.0", "end")
        textbox2.delete("0.0", "end")
        ax.clear()
        file_path = 'Forecast New.xlsx'
        df = pd.read_excel(file_path, sheet_name=variable)
        df.set_index('Time', inplace=True)
        df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()
        df['MA12'] = df['Close'].rolling(window=12, center=True).mean()
        df['CMA'] = (df['MA12'].shift(-1) + df['MA12']) / 2
        df['SF'] = df['Close'] / df['CMA']
        seasonal_index = df['SF'].groupby(df.index.month).mean()

        df['SI'] = df.index.month.map(seasonal_index)
        df['D-Seasonal'] = df['Close'] / df['SI']
        df['D-Trend'] = df['D-Seasonal'].diff()

        autocorrelation = acf(df['D-Trend'].iloc[1:], nlags=36, fft=True)
        for lag in range(1, len(autocorrelation)):
            asterisk = ' *' if autocorrelation[lag] > 0.2603778219616477 else ''
            textbox.insert(customtkinter.END, f'Lag {lag} : {autocorrelation[lag]:.4f}{asterisk}\n')
            if autocorrelation[lag] > 0.2603778219616477:
                if lag < 10:
                    textbox.tag_add('color', f'{lag}.8', f'{lag}.15') 
                    textbox.tag_config('color', foreground='red')
                else:
                    textbox.tag_add('color', f'{lag}.8', f'{lag}.15') 
                    textbox.tag_config('color', foreground='red')

        for lag, value in enumerate(autocorrelation[1:], start=1):
            color = '#fdb9c8' if value > 0.2603778219616477 else '#aec6cf'
            ax.bar(lag, value, color=color)
            
        ax.set_title('ACF')
        ax.set_xlabel('Lag Number')
        ax.set_ylabel('Autocorrelation')
        canvas.draw()

def button_event():
    textbox.delete("0.0", "end")
    textbox2.delete("0.0", "end")
    ax.clear()
    global forcasttype
    global select_model
    global variable
    global weightforcast
    forcasttype = "select_model"
#### Holt-Winters
    if select_model == "Holt-Winter":
        file_path = 'Forecast New Extended.xlsx'
        df = pd.read_excel(file_path, sheet_name=variable)
        pd.set_option('display.max_rows', None)
        df.set_index('Time', inplace=True)
        df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()

        model1 = ExponentialSmoothing(
            df['Close'].dropna(), 
            trend='add',
            seasonal='add',
            seasonal_periods=12
        ).fit()

        forecast = model1.forecast(1)
        combined = pd.concat([model1.fittedvalues, forecast])
        df['Forecast'] = combined

        ax.plot(df['Close'], label='Actual')
        ax.plot(combined, label='Holt-Winter', linestyle='--')
        ax.set_title('Actual vs Forecasted')
        ax.set_xlabel('Time')
        ax.set_ylabel('Close Price')
        ax.legend()
        canvas.draw()
        print(df)
        df[f'ME'] = (df['Close'] - df['Forecast'])
        df[f'MAE'] = np.abs(df[f'ME'])
        df[f'MPE'] = (df[f'ME'] / df['Close']) * 100
        df[f'MAPE'] = np.abs(df[f'MPE'])
        df[f'MSE'] = df[f'ME']**2
        df[f'RMSE'] = np.sqrt(df[f'MSE'])

        textbox2.insert(customtkinter.END, f"Forecast Price \n\n: %.4f\n\n" % df.loc['2023-12-01','Forecast'])
        textbox2.insert(customtkinter.END, "Metrics: Forecast\n\n")
        # textbox2.insert(customtkinter.END, f"ME: %.4f\n" % df[f'ME'].mean())
        # textbox2.insert(customtkinter.END, f"MAE: %.4f\n" % df[f'MAE'].mean())
        # textbox2.insert(customtkinter.END, f"MPE: %.4f\n" % df[f'MPE'].mean())
        # textbox2.insert(customtkinter.END, f"MAPE: %.4f\n" % df[f'MAPE'].mean())
        # textbox2.insert(customtkinter.END, f"MSE: %.4f\n" % df[f'MSE'].mean())
        textbox2.insert(customtkinter.END, f"RMSE: %.4f" % df[f'RMSE'].mean())

    if select_model == "Modified Naïve":
### Modified Naïve
        if weightforcast == "W = 0.1":
            file_path = 'Forecast New Extended.xlsx'
            sheet_name = variable
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            pd.set_option('display.max_rows', None)
            df.set_index('Time', inplace=True)
            df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()
            close_clear = df['Close'].dropna()
            df['MA12'] = close_clear.rolling(window=12, center=True).mean()
            df['CMA'] = (df['MA12'].shift(-1) + df['MA12']) / 2
            df['SF'] = df['Close'] / df['CMA']
            seasonal_index = df['SF'].groupby(df.index.month).mean()
            CMA_RemoveNAN = df['CMA'].dropna()
            cma_count = len(CMA_RemoveNAN)

            df['SI'] = df.index.month.map(seasonal_index)
            df['D-Seasonal'] = df['Close'] / df['SI']
            df['D-Trend'] = df['D-Seasonal'].diff()
            df['Modified W=0.1'] = ((df['D-Trend'].shift(1) - df['D-Trend'].shift(2)) * 0.1) + df['D-Trend'].shift(1)
            df['ModifiedFW=0.1'] = df['D-Seasonal'] + df['Modified W=0.1'] - df['D-Trend']
            df['Forecast'] = df['ModifiedFW=0.1'] * df['SI']
            df.iloc[-1, df.columns.get_loc('Forecast')] = df.loc['2023-11-01','D-Seasonal'] + df.loc['2023-12-01','Modified W=0.1']

            print(df)
            ax.plot(df['Close'], label='Actual')
            ax.plot(df['Forecast'], label='Modified Naïve W=0.1', linestyle='--')
            ax.set_title('Actual vs Forecasted')
            ax.set_xlabel('Time')
            ax.set_ylabel('Close Price')
            ax.legend()
            canvas.draw()

            df[f'ME'] = (df['Close'] - df['Forecast'])
            df[f'MAE'] = np.abs(df[f'ME'])
            df[f'MPE'] = (df[f'ME'] / df['Close']) * 100
            df[f'MAPE'] = np.abs(df[f'MPE'])
            df[f'MSE'] = df[f'ME']**2
            df[f'RMSE'] = np.sqrt(df[f'MSE'])

            textbox2.insert(customtkinter.END, f"Forecast Price \n\n: %.4f\n\n" % df.loc['2023-12-01','Forecast'])
            textbox2.insert(customtkinter.END, "Metrics: Forecast\n\n")
            # textbox2.insert(customtkinter.END, f"ME: %.4f\n" % df[f'ME'].mean())
            # textbox2.insert(customtkinter.END, f"MAE: %.4f\n" % df[f'MAE'].mean())
            # textbox2.insert(customtkinter.END, f"MPE: %.4f\n" % df[f'MPE'].mean())
            # textbox2.insert(customtkinter.END, f"MAPE: %.4f\n" % df[f'MAPE'].mean())
            # textbox2.insert(customtkinter.END, f"MSE: %.4f\n" % df[f'MSE'].mean())
            textbox2.insert(customtkinter.END, f"RMSE: %.4f" % df[f'RMSE'].mean())

        if weightforcast == "W = 0.5":
            file_path = 'Forecast New Extended.xlsx'
            sheet_name = variable
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            pd.set_option('display.max_rows', None)
            df.set_index('Time', inplace=True)
            df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()
            close_clear = df['Close'].dropna()
            df['MA12'] = close_clear.rolling(window=12, center=True).mean()
            df['CMA'] = (df['MA12'].shift(-1) + df['MA12']) / 2
            df['SF'] = df['Close'] / df['CMA']
            seasonal_index = df['SF'].groupby(df.index.month).mean()
            CMA_RemoveNAN = df['CMA'].dropna()
            cma_count = len(CMA_RemoveNAN)

            df['SI'] = df.index.month.map(seasonal_index)
            df['D-Seasonal'] = df['Close'] / df['SI']
            df['D-Trend'] = df['D-Seasonal'].diff()
            df['Modified W=0.5'] = ((df['D-Trend'].shift(1) - df['D-Trend'].shift(2)) * 0.5) + df['D-Trend'].shift(1)
            df['ModifiedFW=0.5'] = df['D-Seasonal'] + df['Modified W=0.5'] - df['D-Trend']
            df['Forecast'] = df['ModifiedFW=0.5'] * df['SI']
            df.iloc[-1, df.columns.get_loc('Forecast')] = df.loc['2023-11-01','D-Seasonal'] + df.loc['2023-12-01','Modified W=0.5']

            print(df)
            ax.plot(df['Close'], label='Actual')
            ax.plot(df['Forecast'], label='Modified Naïve W=0.5', linestyle='--')
            ax.set_title('Actual vs Forecasted')
            ax.set_xlabel('Time')
            ax.set_ylabel('Close Price')
            ax.legend()
            canvas.draw()

            df[f'ME'] = (df['Close'] - df['Forecast'])
            df[f'MAE'] = np.abs(df[f'ME'])
            df[f'MPE'] = (df[f'ME'] / df['Close']) * 100
            df[f'MAPE'] = np.abs(df[f'MPE'])
            df[f'MSE'] = df[f'ME']**2
            df[f'RMSE'] = np.sqrt(df[f'MSE'])

            textbox2.insert(customtkinter.END, f"Forecast Price \n\n: %.4f\n\n" % df.loc['2023-12-01','Forecast'])
            textbox2.insert(customtkinter.END, "Metrics: Forecast\n\n")
            # textbox2.insert(customtkinter.END, f"ME: %.4f\n" % df[f'ME'].mean())
            # textbox2.insert(customtkinter.END, f"MAE: %.4f\n" % df[f'MAE'].mean())
            # textbox2.insert(customtkinter.END, f"MPE: %.4f\n" % df[f'MPE'].mean())
            # textbox2.insert(customtkinter.END, f"MAPE: %.4f\n" % df[f'MAPE'].mean())
            # textbox2.insert(customtkinter.END, f"MSE: %.4f\n" % df[f'MSE'].mean())
            textbox2.insert(customtkinter.END, f"RMSE: %.4f" % df[f'RMSE'].mean())

        if weightforcast == "W = 0.9":
            file_path = 'Forecast New Extended.xlsx'
            sheet_name = variable
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            pd.set_option('display.max_rows', None)
            df.set_index('Time', inplace=True)
            df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()
            close_clear = df['Close'].dropna()
            df['MA12'] = close_clear.rolling(window=12, center=True).mean()
            df['CMA'] = (df['MA12'].shift(-1) + df['MA12']) / 2
            df['SF'] = df['Close'] / df['CMA']
            seasonal_index = df['SF'].groupby(df.index.month).mean()
            CMA_RemoveNAN = df['CMA'].dropna()
            cma_count = len(CMA_RemoveNAN)

            df['SI'] = df.index.month.map(seasonal_index)
            df['D-Seasonal'] = df['Close'] / df['SI']
            df['D-Trend'] = df['D-Seasonal'].diff()
            df['Modified W=0.9'] = ((df['D-Trend'].shift(1) - df['D-Trend'].shift(2)) * 0.9) + df['D-Trend'].shift(1)
            df['ModifiedFW=0.9'] = df['D-Seasonal'] + df['Modified W=0.9'] - df['D-Trend']
            df['Forecast'] = df['ModifiedFW=0.9'] * df['SI']
            df.iloc[-1, df.columns.get_loc('Forecast')] = df.loc['2023-11-01','D-Seasonal'] + df.loc['2023-12-01','Modified W=0.9']

            print(df)
            ax.plot(df['Close'], label='Actual')
            ax.plot(df['Forecast'], label='Modified Naïve W=0.9', linestyle='--')
            ax.set_title('Actual vs Forecasted')
            ax.set_xlabel('Time')
            ax.set_ylabel('Close Price')
            ax.legend()
            canvas.draw()

            df[f'ME'] = (df['Close'] - df['Forecast'])
            df[f'MAE'] = np.abs(df[f'ME'])
            df[f'MPE'] = (df[f'ME'] / df['Close']) * 100
            df[f'MAPE'] = np.abs(df[f'MPE'])
            df[f'MSE'] = df[f'ME']**2
            df[f'RMSE'] = np.sqrt(df[f'MSE'])

            textbox2.insert(customtkinter.END, f"Forecast Price \n\n: %.4f\n\n" % df.loc['2023-12-01','Forecast'])
            textbox2.insert(customtkinter.END, "Metrics: Forecast\n\n")
            # textbox2.insert(customtkinter.END, f"ME: %.4f\n" % df[f'ME'].mean())
            # textbox2.insert(customtkinter.END, f"MAE: %.4f\n" % df[f'MAE'].mean())
            # textbox2.insert(customtkinter.END, f"MPE: %.4f\n" % df[f'MPE'].mean())
            # textbox2.insert(customtkinter.END, f"MAPE: %.4f\n" % df[f'MAPE'].mean())
            # textbox2.insert(customtkinter.END, f"MSE: %.4f\n" % df[f'MSE'].mean())
            textbox2.insert(customtkinter.END, f"RMSE: %.4f" % df[f'RMSE'].mean())

#### Basic Naïve
    if select_model == "Basic Naïve":
        file_path = 'Forecast New Extended.xlsx'
        sheet_name = variable
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        pd.set_option('display.max_rows', None)
        df.set_index('Time', inplace=True)
        df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()
        close_clear = df['Close'].dropna()
        df['MA12'] = close_clear.rolling(window=12, center=True).mean()
        df['CMA'] = (df['MA12'].shift(-1) + df['MA12']) / 2
        df['SF'] = df['Close'] / df['CMA']
        seasonal_index = df['SF'].groupby(df.index.month).mean()
        CMA_RemoveNAN = df['CMA'].dropna()
        cma_count = len(CMA_RemoveNAN)

        df['SI'] = df.index.month.map(seasonal_index)
        df['D-Seasonal'] = df['Close'] / df['SI']
        df['D-Trend'] = df['D-Seasonal'].diff()
        df['Basic Naïve'] = df['D-Trend'].shift(1)
        df['Forecast D-Trend'] = df['D-Seasonal'].shift(1) + df['Basic Naïve']
        df['Forecast'] = df['Forecast D-Trend'] * df['SI']
        print(df)
        ax.plot(df['Close'], label='Actual')
        ax.plot(df['Forecast'], label='Basic Naïve', linestyle='--')
        ax.set_title('Actual vs Forecasted')
        ax.set_xlabel('Time')
        ax.set_ylabel('Close Price')
        ax.legend()
        canvas.draw()
        
        df[f'ME'] = (df['Close'] - df['Forecast'])
        df[f'MAE'] = np.abs(df[f'ME'])
        df[f'MPE'] = (df[f'ME'] / df['Close']) * 100
        df[f'MAPE'] = np.abs(df[f'MPE'])
        df[f'MSE'] = df[f'ME']**2
        df[f'RMSE'] = np.sqrt(df[f'MSE'])

        textbox2.insert(customtkinter.END, f"Forecast Price \n\n: %.4f\n\n" % df.loc['2023-12-01','Forecast'])
        textbox2.insert(customtkinter.END, "Metrics: Forecast\n\n")
        # textbox2.insert(customtkinter.END, f"ME: %.4f\n" % df[f'ME'].mean())
        # textbox2.insert(customtkinter.END, f"MAE: %.4f\n" % df[f'MAE'].mean())
        # textbox2.insert(customtkinter.END, f"MPE: %.4f\n" % df[f'MPE'].mean())
        # textbox2.insert(customtkinter.END, f"MAPE: %.4f\n" % df[f'MAPE'].mean())
        # textbox2.insert(customtkinter.END, f"MSE: %.4f\n" % df[f'MSE'].mean())
        textbox2.insert(customtkinter.END, f"RMSE: %.4f" % df[f'RMSE'].mean())

#### Moving Average
    if select_model == "Moving Average":
        if weightforcast == "MA = 3":
            file_path = 'Forecast New Extended.xlsx'
            sheet_name = variable
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            pd.set_option('display.max_rows', None)
            df.set_index('Time', inplace=True)
            df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()
            close_clear = df['Close'].dropna()
            df['MA12'] = close_clear.rolling(window=12, center=True).mean()
            df['CMA'] = (df['MA12'].shift(-1) + df['MA12']) / 2
            df['SF'] = df['Close'] / df['CMA']
            seasonal_index = df['SF'].groupby(df.index.month).mean()
            CMA_RemoveNAN = df['CMA'].dropna()
            cma_count = len(CMA_RemoveNAN)

            df['SI'] = df.index.month.map(seasonal_index)
            df['D-Seasonal'] = df['Close'] / df['SI']
            df['D-Trend'] = df['D-Seasonal'].diff()
            df['MA3'] = df['D-Trend'].shift(1).rolling(window=3).mean()
            df['Forecast D-Trend'] = df['D-Seasonal'].shift(1) + df['MA3']
            df['Forecast'] = df['Forecast D-Trend'] * df['SI']
            print(df)
            ax.plot(df['Close'], label='Actual')
            ax.plot(df['Forecast'], label='MA3 with D-Seasonal and D-Trend', linestyle='--')
            ax.set_title('Actual vs Forecasted')
            ax.set_xlabel('Time')
            ax.set_ylabel('Close Price')
            ax.legend()
            canvas.draw()

            df[f'ME'] = (df['Close'] - df['Forecast'])
            df[f'MAE'] = np.abs(df[f'ME'])
            df[f'MPE'] = (df[f'ME'] / df['Close']) * 100
            df[f'MAPE'] = np.abs(df[f'MPE'])
            df[f'MSE'] = df[f'ME']**2
            df[f'RMSE'] = np.sqrt(df[f'MSE'])

            textbox2.insert(customtkinter.END, f"Forecast Price \n\n: %.4f\n\n" % df.loc['2023-12-01','Forecast'])
            textbox2.insert(customtkinter.END, "Metrics: Forecast\n\n")
            # textbox2.insert(customtkinter.END, f"ME: %.4f\n" % df[f'ME'].mean())
            # textbox2.insert(customtkinter.END, f"MAE: %.4f\n" % df[f'MAE'].mean())
            # textbox2.insert(customtkinter.END, f"MPE: %.4f\n" % df[f'MPE'].mean())
            # textbox2.insert(customtkinter.END, f"MAPE: %.4f\n" % df[f'MAPE'].mean())
            # textbox2.insert(customtkinter.END, f"MSE: %.4f\n" % df[f'MSE'].mean())
            textbox2.insert(customtkinter.END, f"RMSE: %.4f" % df[f'RMSE'].mean())

        if weightforcast == "MA = 5":
            file_path = 'Forecast New Extended.xlsx'
            sheet_name = variable
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            pd.set_option('display.max_rows', None)
            df.set_index('Time', inplace=True)
            df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()
            close_clear = df['Close'].dropna()
            df['MA12'] = close_clear.rolling(window=12, center=True).mean()
            df['CMA'] = (df['MA12'].shift(-1) + df['MA12']) / 2
            df['SF'] = df['Close'] / df['CMA']
            seasonal_index = df['SF'].groupby(df.index.month).mean()
            CMA_RemoveNAN = df['CMA'].dropna()
            cma_count = len(CMA_RemoveNAN)

            df['SI'] = df.index.month.map(seasonal_index)
            df['D-Seasonal'] = df['Close'] / df['SI']
            df['D-Trend'] = df['D-Seasonal'].diff()
            df['MA5'] = df['D-Trend'].shift(1).rolling(window=5).mean()
            df['Forecast D-Trend'] = df['D-Seasonal'].shift(1) + df['MA5']
            df['Forecast'] = df['Forecast D-Trend'] * df['SI']
            print(df)
            ax.plot(df['Close'], label='Actual')
            ax.plot(df['Forecast'], label='MA5 with D-Seasonal and D-Trend', linestyle='--')
            ax.set_title('Actual vs Forecasted')
            ax.set_xlabel('Time')
            ax.set_ylabel('Close Price')
            ax.legend()
            canvas.draw()

            df[f'ME'] = (df['Close'] - df['Forecast'])
            df[f'MAE'] = np.abs(df[f'ME'])
            df[f'MPE'] = (df[f'ME'] / df['Close']) * 100
            df[f'MAPE'] = np.abs(df[f'MPE'])
            df[f'MSE'] = df[f'ME']**2
            df[f'RMSE'] = np.sqrt(df[f'MSE'])

            textbox2.insert(customtkinter.END, f"Forecast Price \n\n: %.4f\n\n" % df.loc['2023-12-01','Forecast'])
            textbox2.insert(customtkinter.END, "Metrics: Forecast\n\n")
            # textbox2.insert(customtkinter.END, f"ME: %.4f\n" % df[f'ME'].mean())
            # textbox2.insert(customtkinter.END, f"MAE: %.4f\n" % df[f'MAE'].mean())
            # textbox2.insert(customtkinter.END, f"MPE: %.4f\n" % df[f'MPE'].mean())
            # textbox2.insert(customtkinter.END, f"MAPE: %.4f\n" % df[f'MAPE'].mean())
            # textbox2.insert(customtkinter.END, f"MSE: %.4f\n" % df[f'MSE'].mean())
            textbox2.insert(customtkinter.END, f"RMSE: %.4f" % df[f'RMSE'].mean())

        if weightforcast == "MA = 7":
            file_path = 'Forecast New Extended.xlsx'
            sheet_name = variable
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            pd.set_option('display.max_rows', None)
            df.set_index('Time', inplace=True)
            df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()
            close_clear = df['Close'].dropna()
            df['MA12'] = close_clear.rolling(window=12, center=True).mean()
            df['CMA'] = (df['MA12'].shift(-1) + df['MA12']) / 2
            df['SF'] = df['Close'] / df['CMA']
            seasonal_index = df['SF'].groupby(df.index.month).mean()
            CMA_RemoveNAN = df['CMA'].dropna()
            cma_count = len(CMA_RemoveNAN)

            df['SI'] = df.index.month.map(seasonal_index)
            df['D-Seasonal'] = df['Close'] / df['SI']
            df['D-Trend'] = df['D-Seasonal'].diff()
            df['MA7'] = df['D-Trend'].shift(1).rolling(window=7).mean()
            df['Forecast D-Trend'] = df['D-Seasonal'].shift(1) + df['MA7']
            df['Forecast'] = df['Forecast D-Trend'] * df['SI']
            print(df)
            ax.plot(df['Close'], label='Actual')
            ax.plot(df['Forecast'], label='MA7 with D-Seasonal and D-Trend', linestyle='--')
            ax.set_title('Actual vs Forecasted')
            ax.set_xlabel('Time')
            ax.set_ylabel('Close Price')
            ax.legend()
            canvas.draw()

            df[f'ME'] = (df['Close'] - df['Forecast'])
            df[f'MAE'] = np.abs(df[f'ME'])
            df[f'MPE'] = (df[f'ME'] / df['Close']) * 100
            df[f'MAPE'] = np.abs(df[f'MPE'])
            df[f'MSE'] = df[f'ME']**2
            df[f'RMSE'] = np.sqrt(df[f'MSE'])

            textbox2.insert(customtkinter.END, f"Forecast Price \n\n: %.4f\n\n" % df.loc['2023-12-01','Forecast'])
            textbox2.insert(customtkinter.END, "Metrics: Forecast\n\n")
            # textbox2.insert(customtkinter.END, f"ME: %.4f\n" % df[f'ME'].mean())
            # textbox2.insert(customtkinter.END, f"MAE: %.4f\n" % df[f'MAE'].mean())
            # textbox2.insert(customtkinter.END, f"MPE: %.4f\n" % df[f'MPE'].mean())
            # textbox2.insert(customtkinter.END, f"MAPE: %.4f\n" % df[f'MAPE'].mean())
            # textbox2.insert(customtkinter.END, f"MSE: %.4f\n" % df[f'MSE'].mean())
            textbox2.insert(customtkinter.END, f"RMSE: %.4f" % df[f'RMSE'].mean())

### Decomposition
    if select_model == "Decomposition":
        file_path = 'Forecast New.xlsx'
        sheet_name = variable
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        pd.set_option('display.max_rows', None)
        df.set_index('Time', inplace=True)
        df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()
        df['MA12'] = df['Close'].rolling(window=12, center=True).mean()
        df['CMA'] = (df['MA12'].shift(-1) + df['MA12']) / 2
        df['SF'] = df['Close'] / df['CMA']

        seasonal_index = df['SF'].groupby(df.index.month).mean()

        CMA_RemoveNAN = df['CMA'].dropna()
        cma_count = len(CMA_RemoveNAN)
        time_list = list(range(1, cma_count + 1))

        model = sm.OLS(CMA_RemoveNAN, sm.add_constant(time_list)).fit()
        model_summary = model.summary()
        # print(model_summary)

        intercept_var = model.params['const']
        slope_var = model.params['x1']
        # print(intercept_var)
        # print(slope_var)
        df['CMAT'] = intercept_var + (slope_var * pd.Series(time_list, index=CMA_RemoveNAN.index))
        df['CF'] = df['CMA'] / df['CMAT']
        df['SI'] = df.index.month.map(seasonal_index)
        df['Forecast'] = df['CMAT'] * df['CF'] * df['SI']
        print(df)
        ax.plot(df['Close'], label='Actual')
        ax.plot(df['Forecast'], label='Forecasted', linestyle='--')
        ax.set_title('Actual vs Forecasted')
        ax.set_xlabel('Time')
        ax.set_ylabel('Close Price')
        ax.legend()
        canvas.draw()

        df[f'ME'] = (df['Close'] - df['Forecast'])
        df[f'MAE'] = np.abs(df[f'ME'])
        df[f'MPE'] = (df[f'ME'] / df['Close']) * 100
        df[f'MAPE'] = np.abs(df[f'MPE'])
        df[f'MSE'] = df[f'ME']**2
        df[f'RMSE'] = np.sqrt(df[f'MSE'])

        textbox2.insert(customtkinter.END, "Metrics: Forecast\n\n")
        # textbox2.insert(customtkinter.END, f"ME: %.4f\n" % df[f'ME'].mean())
        # textbox2.insert(customtkinter.END, f"MAE: %.4f\n" % df[f'MAE'].mean())
        # textbox2.insert(customtkinter.END, f"MPE: %.4f\n" % df[f'MPE'].mean())
        # textbox2.insert(customtkinter.END, f"MAPE: %.4f\n" % df[f'MAPE'].mean())
        # textbox2.insert(customtkinter.END, f"MSE: %.4f\n" % df[f'MSE'].mean())
        textbox2.insert(customtkinter.END, f"RMSE: %.4f" % df[f'RMSE'].mean())

def checkbox_event():
    global checktick_seasonal
    global forcasttype
    checktick_seasonal = checkbox.get()
    if checktick_seasonal == "on" and forcasttype == "autocorrelation":
        autocorrelation_event()
    if checktick_seasonal == "off" and forcasttype == "autocorrelation":
        autocorrelation_event()

def disable_event():
    app.quit() 
    app.destroy() 

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

app = customtkinter.CTk()
app.geometry("1280x720")
app.resizable(False, False)
app.title("Forecasting Model")

app.protocol("WM_DELETE_WINDOW", disable_event)
options_frame = customtkinter.CTkFrame(app)

options_frame.pack(side=customtkinter.LEFT)
options_frame.pack_propagate(False)
options_frame.configure(width=200, height=720)

options_frame2 = customtkinter.CTkFrame(app)

optionstitle2 = customtkinter.CTkLabel(options_frame2, text="Analysis")
optionstitle2.pack(padx=10,pady=10)

options_frame2.pack(side=customtkinter.RIGHT)
options_frame2.pack_propagate(False)
options_frame2.configure(width=200, height=720)

optionstitle = customtkinter.CTkLabel(options_frame, text="Settings")
optionstitle.pack(padx=10,pady=10)

combobox = customtkinter.CTkComboBox(master=options_frame,values=
                                     ["TESLA", "MICROSOFT", "NASDAQ", "S&P500"], 
                                     width=180, height=32, command=combobox_callback, state='readonly')
combobox.pack(padx=20, pady=10)
combobox.set("TESLA")

button = customtkinter.CTkButton(options_frame, text="Calculate Autocorrelation", width=180, height=32,command=autocorrelation_event)
button.pack(padx=10,pady=10)

checkbox = customtkinter.CTkCheckBox(master=options_frame, text="Seasonality and Trend",command=checkbox_event,onvalue="on", offvalue="off")
checkbox.pack(padx=10, pady=5)

button4 = customtkinter.CTkButton(options_frame, text="Check Seasonality", width=180, height=32,command=seasonal_event)
button4.pack(padx=10,pady=10)

buttonT = customtkinter.CTkButton(options_frame, text="Check Trend", width=180, height=32,command=trend_event)
buttonT.pack(padx=10,pady=10)

button2 = customtkinter.CTkButton(options_frame, text="Calculate Correlation", width=180, height=32,command=correlation_event)
button2.pack(padx=10,pady=10)

textbox = customtkinter.CTkTextbox(options_frame2, height=280)
# textbox.configure(state="disabled")
textbox.pack(padx=10,pady=10)

textbox2 = customtkinter.CTkTextbox(options_frame2)
# textbox2.configure(state="disabled")
textbox2.pack(padx=10,pady=10)

combobox2 = customtkinter.CTkComboBox(master=options_frame,values=[
    "Basic Naïve","Moving Average", "Modified Naïve", "Decomposition","Holt-Winter"], width=180, height=32, state='readonly',command=combobox_callback_model)
combobox2.pack(padx=20, pady=10)
combobox2.set("Basic Naïve")

combobox3 = customtkinter.CTkComboBox(master=options_frame,values=
                                     ["None"], 
                                     width=180, height=32, state='readonly',command=checkweight_event)
combobox3.pack(padx=20, pady=10)
combobox3.set("None")

button3 = customtkinter.CTkButton(options_frame, text="Calculate Forecasting Model", width=180, height=32,command=button_event)
button3.pack(padx=10,pady=10)

fig = Figure(figsize=(10, 6))
ax = fig.add_subplot(111)

canvas = FigureCanvasTkAgg(fig, master=app)
canvas.get_tk_widget().pack(side=customtkinter.TOP, fill=customtkinter.BOTH, expand=1)
app.mainloop()

