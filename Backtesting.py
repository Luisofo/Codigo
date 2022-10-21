import matplotlib.pyplot as plt


def backtest_1day_hold(returns,predictions,verbose=0,show_market=0,label=None):

    cum_ret = 1 # Al empezar hemos ganado 0
    equity = list() # 0 a nuestro dinero
    equity.append(cum_ret)
    i=1
    win=0
    loss=0

    for ret,pred in zip(returns,predictions):
        if pred == 1:
            # Go Long for the day
            amount_at_close = 1.0 * (1.0+ret)
            equity_change = amount_at_close - 1
            cum_ret += equity_change
            if ret>=0:
                win+=1
            else:
                loss+=1
            if(verbose):
                print('Day ',i,'Going long : Amount betted: 1 Daily return:',ret,' End amount:',amount_at_close,'Equity change:',equity_change,'Equity:',cum_ret)

        else:
            # Go Short for the day
            amount_at_close = 1.0 * (1.0+(-1.0*ret))
            equity_change = amount_at_close - 1
            cum_ret += equity_change
            if ret<0:
                win+=1
            else:
                loss+=1
            if(verbose):
                print('Day ',i,'Going Short : Amount betted: 1 Daily return:',ret,' End amount:',amount_at_close,'Equity change:',equity_change,'Equity:',cum_ret)

        equity.append(cum_ret)
        i+=1

    if(show_market):
        market_equity = []
        market_equity.append(1)

        for ret in returns:
            market_equity.append(market_equity[-1]*(1+ret))
        plt.plot(range(len(market_equity)),market_equity,label='S&P500')


    plt.plot(range(i),equity,label=label)
    print('Win rate: ({})'.format(label) ,win/i)



def backtest_1day_hold_train_test(train_returns,train_predictions,test_returns,test_predictions,verbose=0):

    cum_ret = 0 # Al empezar hemos ganado 0
    equity_train = list() # 0 a nuestro dinero
    equity_train.append(cum_ret)
    equity_test = list() # 0 a nuestro dinero
    i=1
    indices_train = []
    indices_train.append(i)
    indices_test = []
    win=0
    loss=0

    for ret,pred in zip(train_returns,train_predictions):
        if pred == 1:
            # Go Long for the day
            amount_at_close = 1.0 * (1.0+ret)
            equity_change = amount_at_close - 1
            cum_ret += equity_change
            if ret>=0:
                win+=1
            else:
                loss+=1
            if(verbose):
                print('Day ',i,'Going long : Amount betted: 1 Daily return:',ret,' End amount:',amount_at_close,'Equity change:',equity_change,'Equity:',cum_ret)

        else:
            # Go Short for the day
            amount_at_close = 1.0 * (1.0+abs(ret))
            equity_change = amount_at_close - 1
            cum_ret += equity_change
            if ret<0:
                win+=1
            else:
                loss+=1
            if(verbose):
                print('Day ',i,'Going Short : Amount betted: 1 Daily return:',ret,' End amount:',amount_at_close,'Equity change:',equity_change,'Equity:',cum_ret)

        equity_train.append(cum_ret)
        i+=1
        indices_train.append(i)

    equity_test.append(equity_train[-1])
    indices_test.append(i)

    for ret,pred in zip(test_returns,test_predictions):
        if pred == 1:
            # Go Long for the day
            amount_at_close = 1.0 * (1.0+ret)
            equity_change = amount_at_close - 1
            cum_ret += equity_change
            if ret>=0:
                win+=1
            else:
                loss+=1
            if(verbose):
                print('Day ',i,'Going long : Amount betted: 1 Daily return:',ret,' End amount:',amount_at_close,'Equity change:',equity_change,'Equity:',cum_ret)

        else:
            # Go Short for the day
            amount_at_close = 1.0 * (1.0+abs(ret))
            equity_change = amount_at_close - 1
            cum_ret += equity_change
            if ret<0:
                win+=1
            else:
                loss+=1
            if(verbose):
                print('Day ',i,'Going Short : Amount betted: 1 Daily return:',ret,' End amount:',amount_at_close,'Equity change:',equity_change,'Equity:',cum_ret)

        equity_test.append(cum_ret)
        i+=1
        indices_test.append(i)


    plt.plot(indices_train,equity_train)
    plt.plot(indices_test,equity_test)



def backtest_1day_hold_train_test_market(train_returns,train_predictions,test_returns,test_predictions,verbose=0):

    cum_ret = 0 # Al empezar hemos ganado 0
    equity_train = list() # 0 a nuestro dinero
    equity_train.append(cum_ret)
    equity_test = list() # 0 a nuestro dinero
    i=1
    indices_train = []
    indices_train.append(i)
    indices_test = []
    win=0
    loss=0

    for ret,pred in zip(train_returns,train_predictions):
        if pred == 1:
            # Go Long for the day
            amount_at_close = 1.0 * (1.0+ret)
            equity_change = amount_at_close - 1
            cum_ret += equity_change
            if ret>=0:
                win+=1
            else:
                loss+=1
            if(verbose):
                print('Day ',i,'Going long : Amount betted: 1 Daily return:',ret,' End amount:',amount_at_close,'Equity change:',equity_change,'Equity:',cum_ret)

        else:
            # Go Short for the day
            amount_at_close = 1.0 * (1.0+abs(ret))
            equity_change = amount_at_close - 1
            cum_ret += equity_change
            if ret<0:
                win+=1
            else:
                loss+=1
            if(verbose):
                print('Day ',i,'Going Short : Amount betted: 1 Daily return:',ret,' End amount:',amount_at_close,'Equity change:',equity_change,'Equity:',cum_ret)

        equity_train.append(cum_ret)
        i+=1
        indices_train.append(i)

    equity_test.append(equity_train[-1])
    indices_test.append(i)

    for ret,pred in zip(test_returns,test_predictions):
        if pred == 1:
            # Go Long for the day
            amount_at_close = 1.0 * (1.0+ret)
            equity_change = amount_at_close - 1
            cum_ret += equity_change
            if ret>=0:
                win+=1
            else:
                loss+=1
            if(verbose):
                print('Day ',i,'Going long : Amount betted: 1 Daily return:',ret,' End amount:',amount_at_close,'Equity change:',equity_change,'Equity:',cum_ret)

        else:
            # Go Short for the day
            amount_at_close = 1.0 * (1.0+abs(ret))
            equity_change = amount_at_close - 1
            cum_ret += equity_change
            if ret<0:
                win+=1
            else:
                loss+=1
            if(verbose):
                print('Day ',i,'Going Short : Amount betted: 1 Daily return:',ret,' End amount:',amount_at_close,'Equity change:',equity_change,'Equity:',cum_ret)

        equity_test.append(cum_ret)
        i+=1
        indices_test.append(i)

    market_equity = []
    market_equity.append(1)

    for ret in train_returns:
        market_equity.append(market_equity[-1]*(1+ret))
    for ret in test_returns:
        market_equity.append(market_equity[-1]*(1+ret))

    plt.plot(indices_train,equity_train)
    plt.plot(indices_test,equity_test)
    plt.plot(range(len(market_equity)),market_equity)

def backtest_1day_hold_open(open,close,returns,predictions,verbose=0,show_market=0,label=None):

    cum_ret = 1 # Al empezar hemos ganado 0
    equity = list() # 0 a nuestro dinero
    equity.append(cum_ret)
    i=1
    win=0
    loss=0

    for open_price,close_price,pred in zip(open,close,predictions):
        if pred == 1:
            # Go Long for the day
            amount_at_close = 1.0 * (close_price/open_price)
            equity_change = amount_at_close - 1
            cum_ret += equity_change
            if equity_change>=0:
                win+=1
            else:
                loss+=1
            if(verbose):
                print('Day ',i,'Going long : Amount betted: 1 Open:',open_price,'Close:',close_price,' End amount:',amount_at_close,
                      'Equity change:',equity_change,'Equity:',cum_ret)

        else:
            # Go Short for the day
            amount_at_close = 1.0 * (1.0+(-1.0*(close_price/open_price)))
            equity_change = amount_at_close - 1
            cum_ret += equity_change
            if equity_change<0:
                win+=1
            else:
                loss+=1
            if(verbose):
                print('Day ', i, 'Going short : Amount betted: 1 Open:', open_price, 'Close:', close_price,
                      ' End amount:', amount_at_close,'Equity change:', equity_change, 'Equity:', cum_ret)

        equity.append(cum_ret)
        i+=1

    if(show_market):
        market_equity = []
        market_equity.append(1)

        for ret in returns:
            market_equity.append(market_equity[-1]*(1+ret))
        plt.plot(range(len(market_equity)),market_equity,label='S&P500')


    plt.plot(range(i),equity,label=label)
    print('Win rate: ({})'.format(label) ,win/i)