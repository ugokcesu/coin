# First import the libraries that we need to use
import pandas as pd
import numpy as np

import requests
import json
import matplotlib.pyplot as plt
import datetime

import cbpro


def fetch_daily_data(symbol,granularity,start,end):
    #pair_split = symbol.split('/')  # symbol must be in format XXX/XXX ie. BTC/EUR
    #symbol = pair_split[0] + '-' + pair_split[1]
    url = f'https://api.pro.coinbase.com/products/{symbol}/candles?granularity={granularity}?start={start}?end={end}'
    response = requests.get(url)
    if response.status_code == 200:  # check to make sure the response from server is good
        data = pd.DataFrame(json.loads(response.text), columns=['unix', 'low', 'high', 'open', 'close', 'volume'])
        data['date'] = pd.to_datetime(data['unix'], unit='s')  # convert to a readable date
        data['vol_fiat'] = data['volume'] * data['close']      # multiply the BTC volume by closing price to approximate fiat volume

        # if we failed to get any data, print an error...otherwise write the file
        if data is None:
            print("Did not return any data from Coinbase for this symbol")
        else:
            return data
            data.to_csv(f'Coinbase_{pair_split[0] + pair_split[1]}_dailydata.csv', index=False)

    else:
        print("Did not receieve OK response from Coinbase API")

def frameData(data):
    df = pd.DataFrame(data, columns=['time','low','high','open', 'close', 'vol'])
    df['vol'] = df['vol'] * df['close']
    df['date'] = pd.to_datetime(df['time'],unit='s')
    df = df.set_index('date').sort_index()
    #df['pct'] = df['close'].pct_change()
    return df

def get_data(product, months=24, granularity=86400):
    MAX_SAMPLES = 300
    seconds = months * 30 * 24 * 60 *60
    samplesNeeded = seconds / granularity
    startTime = (datetime.datetime.now() - datetime.timedelta(seconds = seconds))
    startTime = datetime.datetime(startTime.year, startTime.month, startTime.day)

    if seconds - MAX_SAMPLES * granularity < 0:
        endTime = datetime.datetime.now()
    else:
        endTime = startTime + datetime.timedelta(seconds = (MAX_SAMPLES) * granularity)
    result = []
    for i in range(10):
        #print(startTime.isoformat(), endTime.isoformat())
        cb = cbpro.PublicClient()
        daily = cb.get_product_historic_rates(product, granularity=granularity,start=startTime.isoformat(), end=endTime.isoformat())
        
        #daily = fetch_daily_data(product, granularity, startTime.isoformat(), endTime.isoformat())
        result.append(frameData(daily))
        startTime = endTime + datetime.timedelta(seconds= 1 * granularity) #Neden bunu comment out edince dogru sonuc cikiyo?
        if startTime > datetime.datetime.now():
            break
        endTime = endTime + datetime.timedelta(seconds= (MAX_SAMPLES) * granularity)
        if endTime > datetime.datetime.now():
            endTime = datetime.datetime.now()
    return pd.concat(result, verify_integrity=True)

def gather_products_list():
    cb = cbpro.PublicClient()
    products = cb.get_products()
    symbols = pd.read_csv("symbols.csv")
    symbols['Cap'] = symbols['Cap'].str.replace("$","").str.replace(",","").astype(np.int64)
    file = open("products.txt", 'w')
    for symbol in symbols['Symbol']:
        for product in products:
            if symbol+ "-USD" == product['id']:
                file.write(product['id']+'\n')
    file.close()

def read_products_list(fileName='products.txt'):
    file = open(fileName, 'r')
    return list(map(lambda x: x.replace("\n", ""), list(file)))
    file.close()

def merge_close_prices(granularity=21600, fileName="products.txt"):
    products = read_products_list(fileName=fileName)
    for i, product in enumerate(products):
        if i == 0:
            result = get_data(product, granularity=granularity)[['close']]   
        else:
            result = pd.concat([result,  get_data(product, granularity=granularity)[['close']]], axis=1)
        result.rename(columns={'close':product}, inplace=True)
    return result

def calculate_std_from_mean(df, window=60):
    #window is number of samples to look back while computing mean and standard deviation
    m = df.rolling(window, min_periods=0).mean()
    return ((df - m) /df.rolling(window, min_periods=0).std())

def keeperProducts(df):
    keeperColumns = df.count().sort_values(ascending=False)[:11].index.tolist()
    file = open('products.txt', 'w')
    for col in keeperColumns:
        file.write(col+'\n')
    file.close()

def prior_proportions(df, target, limit):
    pct_change = convert_to_ternary_change(df, target, limit, shift=0)
    prior = pct_change.groupby(target)[target].count()
    prior = prior / prior.sum()
    return prior 

def convert_to_ternary_change(df, target, limit, shift):
    pct_change = df.pct_change()
    pct_change[pct_change < -limit] = -1
    pct_change[pct_change > limit] = 1
    pct_change[(pct_change >=-limit) & (pct_change <= limit)] = 0
    if shift != 0:
        pct_change[target] = pct_change[target].shift(shift)
    pct_change = pct_change.dropna()
    return pct_change

def get_indexes_before_one(close_prices, target, limit):
    ternary = convert_to_ternary_change(close_prices, target, limit, -1)
    return ternary[ternary[target]==1.0].index

def counted_proportions(df, driver, target, shift=-1, limit=0.02):
    pct_change = convert_to_ternary_change(df, target, limit, shift=shift)
    
    etc = pct_change.groupby(driver)[target].value_counts().to_frame()
    etc.rename({target:'counts'}, axis=1, inplace=True)
    etc = etc.reset_index()
    etc['count prop'] = etc['counts'] / etc.groupby(driver)['counts'].transform('sum')

    columns = ['likelihood-1', 'likelihood0', 'likelihood1']
    etc['likelihood-1'] = etc[etc[target]==-1]['counts'] / pct_change[pct_change[target]==-1][target].size
    etc['likelihood0'] = etc[etc[target]==0]['counts'] / pct_change[pct_change[target]==0][target].size
    etc['likelihood1'] = etc[etc[target]==1]['counts'] / pct_change[pct_change[target]==1][target].size
    etc['likelihood'] = etc[columns].sum(1)
    etc = etc.drop(columns, 1)
    return etc
    
def counter_proportions_dictionary(df, target, products, shift=-1, limit=0.02):
    result = {}

    for product in products:
        if product == target:
            continue
        result[product] = counted_proportions(df, product, target, shift=shift, limit=limit)
    return result

def count_inversions(serie):
    inversions = 0
    for i in range(serie.size):
        for j in range(i+1, serie.size):
            if (serie[i] < serie[j]):
                inversions += 1
    return inversions


def predictor(df_prices, indexesBeforeOne, target, products, limit ) :
   
    prior = prior_proportions(df_prices, target, limit)
    #print("PRIOR: ")
    #print(prior)
    likelihood = counter_proportions_dictionary(df_prices, target, products,shift=-1, limit=limit)
    
    result = []
    ternary = convert_to_ternary_change(df_prices, 'None', limit, 0)
    for index in indexesBeforeOne:
        pmf = Pmf(prior.copy(),target)
        for product in products:
            if product==target:
                continue
            #print("product: " + product +" = " + str(row[product]))
            #x = likelihood[product]
            #print(x[x[product]==row[product]])
            pmf.update(likelihood, product, ternary.loc[index][product])
        result.append(pmf.h[1.0])
    return np.array(result)

class Pmf:
    def __init__(self, h, target):
        self.h = h
        self.target = target
    
    def update(self, likelihood, driver, observed_val):
        x = likelihood[driver]
        for val in [-1, 0, 1]:
            like = x[ (x[driver]== observed_val) & (x[self.target] == val) ]['likelihood']
            self.h[val] *= like
        self.h = self.h/self.h.sum()

    
    def print(self):
        print(self.h)













if __name__ == "__main__":
    # we set which pair we want to retrieve data for
    pair = "BTC/USD"
    fetch_daily_data(symbol=pair)


              