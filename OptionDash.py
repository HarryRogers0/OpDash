import streamlit as st
import yfinance as yf 
import pandas as pd
import numpy as np
import datetime as dt
import altair as alt
from scipy.stats import norm

def create_bar_chart(data, x_column, y_column, color_column=None, title=None):
    """
    Create a bar chart using Altair.

    Parameters:
    - data (DataFrame): Input data for the chart.
    - x_column (str): Name of the column for the x-axis.
    - y_column (str): Name of the column for the y-axis.
    - color_column (str, optional): Name of the column for coloring bars (default None).
    - title (str, optional): Title of the chart (default None).

    Returns:
    - alt.Chart: The Altair bar chart object.
    """
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X(x_column, title=x_column),
        y=alt.Y(y_column, title=y_column),
    ).properties(
        width=600,
        height=400,
    )

    return chart

def create_grouped_bar_chart(data, x, y1, y2, title):
    df = pd.DataFrame(data)
    # Melt the DataFrame to long format for Altair visualization
    df_melted = df.melt(id_vars=[x], value_vars=[y1, y2], var_name='Variable', value_name='Value')
   
    # Create the bar chart using Altair
    chart = alt.Chart(df_melted).mark_bar().encode(
        x=f'{x}:O',  # Use Ordinal scale for categorical x-axis
        y=alt.Y('Value', title='Value'),  # Quantitative scale for y-axis
        color='Variable:N',  # Color bars based on the variable
       #column='Variable:N'  # Separate bars for Value1 and Value2
        xOffset ='Variable:N'
    ).properties(
        width=600,
        height=400,
        title=title
    ).configure_view(
        stroke = None,
    )
    
    return chart


def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the fair price of a European call or put option using the Black-Scholes formula.

    Parameters:
    S (float): Current price of the underlying asset
    K (float): Strike price of the option
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized, expressed as a decimal)
    sigma (float): Volatility of the underlying asset (annualized, expressed as a decimal)
    option_type (str): Type of option, either 'call' or 'put'

    Returns:
    float: Fair price of the option
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be 'call' or 'put'.")

    return option_price

def American_Binomial(S, K_list, T, r, sigma, steps, option_type='call'):
    """
    Calculate the fair prices of American options for multiple strike prices using the binomial tree method.

    Parameters:
    S (float): Current price of the underlying asset
    K_list (list or array): List of strike prices for which option valuations are needed
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate (annualized, expressed as a decimal)
    sigma (float): Volatility of the underlying asset (annualized, expressed as a decimal)
    steps (int): Number of time steps in the binomial tree
    option_type (str): Type of option, either 'call' or 'put'

    Returns:
    list: Fair prices of the options corresponding to each strike price in K_list
    """
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    option_values_list = []  # List to store option values for each strike price

    for K in K_list:
        # Initialize option values at maturity
        prices = np.zeros(steps + 1)
        option_values = np.zeros(steps + 1)

        for i in range(steps + 1):
            prices[i] = S * (u ** (steps - i)) * (d ** i)
            option_values[i] = call_payoff(prices[i], K) if option_type == 'call' else put_payoff(prices[i], K)

        # Calculate option values at earlier nodes
        for j in range(steps - 1, -1, -1):
            prices = prices[:-1] / u
            option_values = (p * option_values[:-1] + (1 - p) * option_values[1:]) * np.exp(-r * dt)

            if option_type == 'call':
                option_values = np.maximum(option_values, call_payoff(prices, K))
            else:
                option_values = np.maximum(option_values, put_payoff(prices, K))

        option_values_list.append(option_values[0])  # Store the option value for this strike price

    return option_values_list

def call_payoff(S, K):
    return np.maximum(0, S - K)

def put_payoff(S, K):
    return np.maximum(0, K - S)



def get_current_price(Symbol):
    todays_data = Symbol.history(period='1d')
    return todays_data['Close'][0]


st.title("Options Analysis Dashboard")
st.write("A Basic Dashboard that when given a Ticker and American Option Expiry will calculate the Black-Scholes-Merton price of the option as well as the binomial tree price.")
ticker = st.text_input("ticker", value = "SPY")
Symbol = yf.Ticker(ticker)
# Download historical data in order to find volatility
historicals_5y = Symbol.history(period="5y")
historicals_1y = Symbol.history(period="1y")
st.write("Stock Price over 5 Years")
st.line_chart(historicals_5y['Close'])

expiry_data = Symbol.options
expiry = st.selectbox("Select an expiry ", expiry_data)

opt = Symbol.option_chain(expiry)

# Initialise dataset
calls = pd.DataFrame(opt.calls)[["strike", "bid", "ask", "openInterest", "impliedVolatility", "inTheMoney"]]
puts = pd.DataFrame(opt.puts)[["strike", "bid", "ask", "openInterest", "impliedVolatility", "inTheMoney"]]
calls = calls.rename(columns = {"bid" : "Bid", "ask": "Ask", "impliedVolatility" : "IV", "inTheMoney" : "ITM", "openInterest" : "Open Interest"})
puts = puts.rename(columns = {"bid" : "Bid", "ask": "Ask", "impliedVolatility" : "IV", "inTheMoney" : "ITM", "openInterest" : "Open Interest"})

st.write(f'## Option Chain for {ticker} - Expiry: {expiry}')

# Displays Option Chains
col1,col2 = st.columns(2)
col1.write("Call Option Chain")
col1.write(calls)
col2.write("Put Option Chain")
col2.write(puts)


# Derivative Pricing
st.write("## Derivative Pricing Basic Models")
#Decision1 = st.selectbox("Put or Call? ", ["Put", "Call"])
Decision1 = st.radio("Put or Call?", ('Put', 'Call'), horizontal=True)
Decision2 = st.radio("Analyse against Black-Scholes-Merton or Binomial Tree", ('Black-Scholes-Merton', 'Binomial Tree'))

# Calculate daily returns
historicals_1y['Daily Return'] = historicals_1y['Close'].pct_change()

# Calculate annualized volatility
volatility = historicals_1y['Daily Return'].std() * np.sqrt(252)  # 252 trading days in a year

#st.write(volatility)


if Decision1 == "Put":
    option_type = "put"
    K = puts['strike']
    actual_ask = puts['Ask']
    
elif Decision1 == "Call":
    option_type = "call"
    K = calls['strike']
    actual_ask = calls['Ask']
    
# American Options Price Calculations
T = (pd.to_datetime(expiry) - pd.to_datetime('today')).days / 365
r = st.slider('Select an annualised risk free interest rate (%)', min_value=0.0, max_value=20.0, value=8.0, step=0.25)/100
Bin_Steps = st.slider('Select the amount of binomial tree steps', min_value=50, max_value=1000, value=150, step=1)
S = get_current_price(Symbol)
Am_BSM = black_scholes(S, K, T, r, volatility, option_type)
#Am_BSM = pd.concat([K, Am_BSM], axis = 1)
Am_Bin = American_Binomial(S, K, T, r, volatility, Bin_Steps, option_type)


Options_Analysis = pd.DataFrame()
Options_Analysis['Strike'] = K
Options_Analysis['BSM'] = Am_BSM
Options_Analysis['Binomial Model'] = Am_Bin
Options_Analysis['Actual Price'] = actual_ask
Options_Analysis['Absolute BSM Error Percentage'] = 100 * abs(Options_Analysis['BSM'] - Options_Analysis['Actual Price']) / Options_Analysis['Actual Price']
Options_Analysis['Absolute Binomial Error Percentage'] = 100 * abs(Options_Analysis['Binomial Model'] - Options_Analysis['Actual Price']) / Options_Analysis['Actual Price']

#st.write(Am_BSM)
#st.write(S)
#st.write(K)
#st.write(Options_Analysis)

if Decision2 == 'Black-Scholes-Merton':
    Options_Analysis = Options_Analysis.query('`Absolute BSM Error Percentage` < 25')
    bar_chart = create_grouped_bar_chart(Options_Analysis, 'Strike', 'Actual Price', 'BSM', 'BSM vs Actual Price')
    bar_chart2 = create_bar_chart(Options_Analysis, 'Strike', 'Absolute BSM Error Percentage')
    MSE = np.square(np.subtract(Options_Analysis['Actual Price'],Options_Analysis['BSM'])).mean()

else:
    Options_Analysis = Options_Analysis.query('`Absolute Binomial Error Percentage` < 25')
    bar_chart = create_grouped_bar_chart(Options_Analysis, 'Strike', 'Actual Price', 'Binomial Model', 'Binomial Model vs Actual Price')
    bar_chart2 = create_bar_chart(Options_Analysis, 'Strike', 'Absolute Binomial Error Percentage')
    MSE = np.square(np.subtract(Options_Analysis['Actual Price'],Options_Analysis['Binomial Model'])).mean()
    
st.altair_chart(bar_chart, use_container_width=True)
st.altair_chart(bar_chart2, use_container_width=True)

st.write(f"The Mean-Square Error of this Option Modelling is {MSE}")

st.write("This dashboard demonstrates how options pricing occurs, and provides a very limited example for both Binomial and Black-Scholes derivative modelling. Further ways the model could be improved would be to create a bespoke volatility model, allowing greater accuracy in the BSM. The model does not currently factor in any Dividends, and assumes constant volatility. If you wish to explore further projects of mine, [please visit my Github page.](https://harryrogers0.github.io)")