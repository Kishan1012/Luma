import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

start_date = '2020-01-01'  
end_date = '2023-01-01'    
tickers = ['SPY', 'NVDA'] 
allocations = [0.6, 0.4]   
portfolio_value = 100000    
time_horizon = 10      
num_simulations = 10

def monte_carlo_simulation(ticker, start_date, end_date, num_simulations=1, time_horizon=252):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    if len(stock_data) < time_horizon:
        print(f"Warning: Only {len(stock_data)} trading days available. Adjusting time horizon to available data.")
        time_horizon = len(stock_data)

    returns = stock_data['Adj Close'].pct_change().dropna()

    last_price = stock_data['Adj Close'][-1]
    mean_return = returns.mean()
    volatility = returns.std()

    simulation_results = np.zeros((time_horizon, num_simulations))
    for i in range(num_simulations):
        daily_returns = np.random.normal(mean_return, volatility, time_horizon) + 1
        price_path = np.zeros(time_horizon)
        price_path[0] = last_price
        for t in range(1, time_horizon):
            price_path[t] = price_path[t - 1] * daily_returns[t]
        simulation_results[:, i] = price_path

    return simulation_results, stock_data.index[:time_horizon]  

def monte_carlo_portfolio_simulation(start_date, end_date, tickers, allocations, portfolio_value, num_simulations=1, time_horizon=252):
    all_simulations = []
    dates = None

    # Run simulations by simulation number first, then ticker
    for sim_num in range(num_simulations):
        simulation_data = []
        
        # Generate and collect simulations for each ticker in this simulation
        for i, ticker in enumerate(tickers):
            price_paths, stock_dates = monte_carlo_simulation(ticker, start_date, end_date, num_simulations=1, time_horizon=time_horizon)
            dates = stock_dates  # Set dates reference from stock_dates of the current ticker
            
            
            df = pd.DataFrame({
                'Date': dates,
                'Simulation': sim_num + 1,
                'Ticker': ticker,
                'Price_Path': price_paths[:, 0]
            })
            simulation_data.append(df)

        
        all_simulations.append(pd.concat(simulation_data))

    
    full_simulations_df = pd.concat(all_simulations).reset_index(drop=True)
    
    print("\nFull Simulation Results:")
    print(full_simulations_df)

    full_simulations_df.to_excel('stacked_simulations_results_without_portfolio.xlsx', sheet_name='All_Simulations')

    print("Simulation results saved to 'stacked_simulations_results_without_portfolio.xlsx'")

monte_carlo_portfolio_simulation(start_date, end_date, tickers, allocations, portfolio_value, num_simulations, time_horizon)
