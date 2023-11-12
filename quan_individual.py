# Import necessary libraries
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import requests
from bs4 import BeautifulSoup
import time

#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")

    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations:
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics,"
                         "majorHoldersBreakdown," # shareholdersInfo
                         "insiderHolders")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret

#==============================================================================
# Header
#==============================================================================
st.title("MY FINANCIAL DASHBOARD ⭐")
col1, col2 = st.columns([1, 5])
col1.write("Data source:")
col2.image('./img/yahoo_finance.png', width=100)

st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio("Go to", ["Summary", "Chart", "Financials", "Monte Carlo", "Analyst Forecast"])

ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
ticker = st.sidebar.selectbox("Ticker", ticker_list)
update_data = st.sidebar.button("Update Data")

from requests.exceptions import HTTPError, RequestException

@st.cache_data
def get_stock_info(ticker):
    stock = YFinance(ticker)
    max_retries = 3

    for attempt in range(max_retries):
        try:
            info = stock.info
            if info:
                return info
        except (HTTPError, RequestException) as e:
            st.warning(f"Attempt {attempt + 1}/{max_retries} failed. Retrying... Error: {e}")
            time.sleep(1)  # Add a delay between retries
            continue
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            break

    st.error(f"Failed to fetch data for {ticker} after {max_retries} attempts.")
    return {}

# Tab 1: Summary
def render_summary_tab():
    st.title("Summary")

    if update_data:
        stock_info = get_stock_info(ticker)

        # Check if stock_info is not None before accessing its elements
        if stock_info:
            # Extracting the company name from the 'longBusinessSummary' key
            company_name = stock_info.get('longBusinessSummary', 'N/A').split(',')[0]
            st.subheader(f"Company Name: {company_name.strip()}")
            st.write(f"Ticker Symbol: {ticker}")
            st.write(f"Sector: {stock_info.get('sector', 'N/A')}")
            st.write(f"Market Cap: {stock_info.get('marketCap', 'N/A')}")
            st.write(f"Previous Close Price: {stock_info.get('regularMarketPreviousClose', 'N/A')}")
            st.write(f"Open Price: {stock_info.get('regularMarketOpen', 'N/A')}")

            # Dropdown for selecting time intervals
            st.subheader("Stock Price Chart")
            selected_interval = st.selectbox("Select Time Interval", ["1D", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"])
            start_date, end_date = get_date_range(selected_interval)
            
            # Display stock price chart
            stock_data = get_stock_price_data(ticker, start_date, end_date)
            fig = go.Figure(data=[go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close']
            )])
            st.plotly_chart(fig)

        # Display company profile, description, and major shareholders
        st.subheader("Company Profile")
        st.write(stock_info.get('longBusinessSummary', 'N/A'))

        # Display major shareholders from yahoo_modules
        st.subheader("Major Shareholders Breakdown")
        # Fetch major holders, institutional holders, and mutual fund holders
        major_holders = yf.Ticker(ticker).major_holders
        institutional_holders = yf.Ticker(ticker).institutional_holders
        mutual_fund_holders = yf.Ticker(ticker).mutualfund_holders
        
        # Display shareholders information
        st.write("**Major Holders**")
        st.dataframe(major_holders)
    
        st.write("**Institutional Holders**")
        st.dataframe(institutional_holders)
    
        st.write("**Mutual Fund Holders**")
        st.dataframe(mutual_fund_holders)

# Helper function to get date range based on selected interval
def get_date_range(selected_interval):
    end_date = datetime.today().date()
    if selected_interval == "1D":
        start_date = end_date - timedelta(days=1)
    elif selected_interval == "1M":
        start_date = end_date - timedelta(days=30)
    elif selected_interval == "3M":
        start_date = end_date - timedelta(days=90)
    elif selected_interval == "6M":
        start_date = end_date - timedelta(days=180)
    elif selected_interval == "YTD":
        start_date = datetime(end_date.year, 1, 1).date()
    elif selected_interval == "1Y":
        start_date = end_date - timedelta(days=365)
    elif selected_interval == "3Y":
        start_date = end_date - timedelta(days=3 * 365)
    elif selected_interval == "5Y":
        start_date = end_date - timedelta(days=5 * 365)
    else:  # MAX
        start_date = datetime(1970, 1, 1).date()
    return start_date, end_date

# Tab 2: Chart
def render_chart_tab():
    st.title("Chart")

    if update_data:
        stock_data = get_stock_price_data(ticker, start_date, end_date)
        st.subheader("Stock Price Chart")
        fig = go.Figure(data=[go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close']
        )])
        st.plotly_chart(fig)

# Function to get stock price data from Yahoo Finance
@st.cache_data
def get_stock_price_data(ticker, start_date, end_date):
    stock_df = yf.Ticker(ticker).history(start=start_date, end=end_date)
    return stock_df

# Tab 3: Financials
def render_financials_tab():
    st.title("Financials")
    statement_type = st.sidebar.selectbox("Select Statement Type", ["Income Statement", "Balance Sheet", "Cash Flow"])
    period = st.sidebar.selectbox("Select Period", ["Annual", "Quarterly"])

    if update_data:
        financial_data = get_financials(ticker, period, statement_type)
        st.subheader(f"{statement_type} ({period})")
        st.write(financial_data)

# Function to get financial data from Yahoo Finance
@st.cache_data
def get_financials(ticker, period, statement_type):
    stock = yf.Ticker(ticker)
    if statement_type == "Income Statement":
        data = stock.financials
    elif statement_type == "Balance Sheet":
        data = stock.balance_sheet
    elif statement_type == "Cash Flow":
        data = stock.cashflow
    if period == "Quarterly":
        data = data.fillna(0).T
    return data

# Tab 4: Monte Carlo simulation
def render_monte_carlo_tab(start_date, end_date):
    st.title("Monte Carlo Simulation")
    n_simulations = st.sidebar.selectbox("Number of Simulations", [200, 500, 1000])
    time_horizon = st.sidebar.selectbox("Time Horizon (days)", [30, 60, 90])

    if update_data:
        results, VaR, simulation_chart = run_monte_carlo_simulation(ticker, n_simulations, time_horizon, start_date, end_date)
        st.subheader("Simulation Results")
        st.write(f"Value at Risk (VaR) at 95% confidence interval: {VaR}")
        st.plotly_chart(simulation_chart)

# Function to run a Monte Carlo simulation for stock closing prices
@st.cache_data
def run_monte_carlo_simulation(ticker, n_simulations, time_horizon, start_date, end_date):
    # Get historical stock price data
    stock_data = get_stock_price_data(ticker, start_date, end_date)

    # Calculate daily returns
    daily_returns = stock_data['Close'].pct_change().dropna()

    # Calculate mean and standard deviation of daily returns
    mean_return = daily_returns.mean()
    std_dev_return = daily_returns.std()

    # Initialize an array to store simulation results
    simulation_results = np.zeros((time_horizon, n_simulations))

    # Generate Monte Carlo simulations
    for i in range(n_simulations):
        # Generate random daily returns based on normal distribution
        daily_return_sim = np.random.normal(mean_return, std_dev_return, time_horizon)

        # Calculate cumulative return
        cumulative_return_sim = np.cumprod(1 + daily_return_sim) - 1

        # Store simulation results in the array
        simulation_results[:, i] = cumulative_return_sim

    # Calculate Value at Risk (VaR) at 95% confidence interval
    VaR = np.percentile(simulation_results[-1, :], 5)

    # Create a plot for the simulation results
    simulation_chart = go.Figure()
    for i in range(n_simulations):
        simulation_chart.add_trace(go.Scatter(x=np.arange(time_horizon), y=simulation_results[:, i], mode='lines', name=f'Simulation {i+1}'))

    simulation_chart.update_layout(title=f'Monte Carlo Simulation for {ticker} Closing Prices',
                                   xaxis_title='Time Horizon (days)',
                                   yaxis_title='Cumulative Return',
                                   showlegend=True)

    return simulation_results, VaR, simulation_chart


# Tab 5: Analyst Forecast
def render_analyst_forecast_tab():
    st.title("Analyst Forecast")

    if update_data:
        analyst_forecast_url = f"https://stockanalysis.com/stocks/{ticker.lower()}/forecast/"

        # Fetch HTML content
        response = requests.get(analyst_forecast_url)
        print(response.text)  # Print HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        print(soup)  # Print the soup object
        print(soup.prettify())  # Print the prettified HTML
        
        # Extract company name and stock price forecast
        company_name_element = soup.find('h1', class_='company')
        stock_price_forecast_element = soup.find('div', class_='stock-price-forecast')
        analyst_ratings_element = soup.find('div', class_='analyst-ratings')
        # Check if elements are found before accessing text attribute
        if company_name_element:
            company_name = company_name_element.text.strip()
        else:
            company_name = "N/A"

        if stock_price_forecast_element:
            stock_price_forecast = stock_price_forecast_element.text.strip()
        else:
            stock_price_forecast = "N/A"
        if analyst_ratings_element:
            analyst_ratings = analyst_ratings_element.text.strip()
        else:
            analyst_ratings = "N/A"
        # Display the information along with the link to open in the browser
        st.write(f"Company Name: {company_name}")
        st.write(f"Stock Price Forecast: {stock_price_forecast}")
        st.write(f"Analyst Ratings: {analyst_ratings}")
        st.markdown(f"Open [Analyst Forecast]( {analyst_forecast_url} ) in your browser.")



# Main Streamlit app
start_date = st.sidebar.date_input("Start date", datetime.today().date() - timedelta(days=30))
end_date = st.sidebar.date_input("End date", datetime.today().date())

if selected_tab == "Summary":
    render_summary_tab()
elif selected_tab == "Chart":
    render_chart_tab()
elif selected_tab == "Financials":
    render_financials_tab()
elif selected_tab == "Monte Carlo":
    render_monte_carlo_tab(start_date, end_date)  # Pass start_date and end_date
elif selected_tab == "Analyst Forecast":
    render_analyst_forecast_tab()

