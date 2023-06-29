from Secret import const
from python_bitvavo_api.bitvavo import Bitvavo
import datetime
import os.path
import smtplib
from email.mime.text import MIMEText
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

bitvavo_info = Bitvavo({
    'APIKEY': const.api_key1,
    'APISECRET': const.api_secret1,
    'RESTURL': 'https://api.bitvavo.com/v2',
    'WSURL': 'wss://ws.bitvavo.com/v2/',
    'ACCESSWINDOW': 10000,
    'DEBUGGING': False
})

bitvavo_action = Bitvavo({
    'APIKEY': const.api_key2,
    'APISECRET': const.api_secret2,
    'RESTURL': 'https://api.bitvavo.com/v2',
    'WSURL': 'wss://ws.bitvavo.com/v2/',
    'ACCESSWINDOW': 10000,
    'DEBUGGING': False
})


def price_list(symbol: str = 'ETH'):
    """
    Convert Bitvavo trading data into a dataframe
    and formats it into an use able dataframe
    Output is df,
    """
    pair = str.upper(symbol) + '-EUR'   # determine pair
    # query Bitvavo database
    resp = bitvavo_info.candles(pair, '1h', {'limit': 48})
    df = pd.DataFrame(resp, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Vol.'])
    df['datetime'] = pd.to_datetime(df['Date'], unit='ms')  # convert date format to datetime
    df.drop(['Date', 'Vol.'], axis=1, inplace=True)  # drop the dead weight
    df = df.sort_values(by='datetime')  # sort values by date
    df_datelist = list(df['datetime'])  # Extract dates
    df = pd.DataFrame(df, columns=['Close', 'High', 'Low', 'Open'])
    df.index = df_datelist

    if df.isnull().sum().sum() > 0 or len(df) < 0:
        print(df.shape)
        print(df.isnull().sum())
    else:
        print('No missing values is dataframe')

    return df


def forecasting(model,
                input_dataset,
                n_steps: int = 48):
    """
    Takes crypto data DataFrame to forecast price development for the coming `number_of_time_points` timepoints.
    It outputs the forecast in a pd.DataFrame array.
    """

    input_df = input_dataset.copy()
    X_timedates = list(input_df[-n_steps:].index)
    X_input = np.array(input_df[-n_steps:])
    X_input = StandardScaler().fit_transform(X_input)
    X_input = X_input.reshape((1, X_input.shape[0], X_input.shape[1]))   # reshape input

    # append new predictions
    y_output = []
    for i in range(0, n_steps):
        # verbose
        print('\r    \r', end='', flush=True)
        print(f'{((i + 1) * (100 / n_steps)):.2f}%', end='', flush=True)

        # adding forecasts
        y_forecast = model.predict(X_input, verbose=0)  # predict next item
        y_output.append(list(StandardScaler().inverse_transform(y_forecast)))
        # add to X_input
        y_forecast = y_forecast.reshape(1, y_forecast.shape[0], y_forecast.shape[1])  # reshape prediction
        X_input = np.hstack((X_input[:, 1:], y_forecast))  # add new prediction to pre-existing input-data
    print()

    # turn output into pd.DataFrame
    if X_timedates:
        for i in range(0, n_steps):
            X_timedates.append(
                X_timedates[-1] + pd.Timedelta("1 hour"))  # add 1 hour to last time point per cycle

        # turn into pd.DataFrame with time-indexes
        y_output = np.array(y_output).astype(float).round(decimals=1)
        y_output = y_output.reshape(y_output.shape[0], y_output.shape[2])
        y_output = pd.DataFrame(np.array(y_output),
                                columns=['Close', 'High', 'Low', 'Open'],
                                index=X_timedates[-n_steps:])  # turn into dataframe
    return y_output


def get_balance(symbol: str):
    try:
        return float(bitvavo_info.balance({"symbol": str.upper(symbol)})[0]['available'])
    except Exception as error:
        log(f'ERROR BALANCE,{symbol},NaN,NaN,NaN,{error}')
        send_mail(action='Error', stringer=f'GET_BALANCE went wrong: {error}')
        print(error)


def trade_market_order(coin: str, delta_ma: float, balance_euro: float, balance_coin: float, price_coin: float,
                       threshold: float,):
    pair = str.upper(coin) + '-EUR'
    action = 'Nothing'
    err = 'none'
    if (delta_ma > threshold) & (balance_euro > 1):  # buy coins with euros
        action = 'Buy'
        try:
            bitvavo_action.placeOrder(pair, 'buy', 'market', {'amountQuote': balance_euro})
        except Exception as error:
            print(error)
            err = error
            send_mail(action='Error', stringer=f'Trade went wrong: {error}')
    elif (delta_ma < 0 - threshold) & (balance_coin > 0.001):  # sell coins for euros
        action = 'Sell'
        try:
            bitvavo_action.placeOrder(pair, 'sell', 'market', {'amount': balance_coin})
        except Exception as error:
            print(error)
            err = error
            send_mail(action='Error', stringer=f'Trade went wrong: {error}')
    stringer = f'\tAction: {action} {pair}\n\tBalance EURO: {balance_euro}\n\tBalance {coin}: {balance_coin}\n'
    stringer += f'\tPrice coin: {price_coin}\n\tDelta_ma: {delta_ma}\n\tError: {err}\n\tTime: {datetime.datetime.now()}'

    send_mail(action=action, stringer=stringer)
    log(f'{action},{pair},{balance_euro},{balance_coin},{delta_ma},{err}')

    return stringer


def log(stringer: str):
    path = ''
    file = f'{path}log.csv'
    text = f'{stringer},{datetime.datetime.now()}\n'
    if os.path.isfile(file):
        with open(file, 'a') as f:
            f.write(text)
            f.close()
    else:
        with open(file, 'w') as g:
            g.write('Action,Pair,Balance_euro,Balance_coin,Delta_MA,Error,DateTime\n' + text)
            g.close()


def send_mail(action: str, stringer: str):
    if (action.lower() == 'buy') or (action.lower() == 'sell') or (action.lower() == 'error'):
        try:
            msg = MIMEText(stringer)
            msg['Subject'] = f'Bitvavo trade {action}'
            msg['From'] = const.email_sender
            msg['To'] = const.email_receiver

            my_mail = smtplib.SMTP('smtp.gmail.com', 587)
            my_mail.ehlo()
            my_mail.starttls()
            my_mail.login(const.email_sender, const.email_sender_password)
            my_mail.sendmail(const.email_sender, const.email_receiver, msg.as_string())
            my_mail.close()
            print("Mail send successfully.")

        except Exception as error:
            log(f'ERROR EMAIL,{action},NaN,NaN,NaN,{error}')
            print(f"something went wrong while trying to send the mail: {error}")
