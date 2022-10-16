"""
This file implements most alphas of the paper
101 Alphas by Zura Kakushadze
https://arxiv.org/ftp/arxiv/papers/1601/1601.00991.pdf

February 2021
Jan Gobeli
"""

import utils as u
import pandas as pd
import numpy as np

####################################ALFAS DISPONIBLES#########################################
def alpha1(df):
    """
    Alpha#1
    (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : Close), 2.), 5)) - 0.5)

    :param df: dataframe
    :return:
    """
    temp1 = pd.Series(np.where((df.returns < 0), u.stddev(df.returns, 20), df.Close), index=df.index)
    return (u.rank(u.ts_argmax(temp1 ** 2, 5)) - 0.5)


def alpha2(df):
    """
    Alpha#2
    (-1 * correlation(rank(delta(log(Volume), 2)), rank(((Close - Open) / Open)), 6))
    """
    tmp_1 = u.rank(u.delta(np.log(df.Volume), 2))
    tmp_2 = u.rank(((df.Close - df.Open) / df.Open))
    return (-1 * u.corr(tmp_1, tmp_2, 6))


def alpha3(df):
    """
    Alpha#3
    (-1 * correlation(rank(Open), rank(Volume), 10))
    """
    return (-1 * u.corr(u.rank(df.Open), u.rank(df.Volume), 10))


def alpha4(df):
    """
    Alpha #4
    (-1 * Ts_Rank(rank(Low), 9))
    """
    return (-1 * u.ts_rank(u.rank(df.Low), 9))


def alpha5(df):
    """
    Alpha#5
    (rank((Open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((Close - vwap)))))
    """
    return (u.rank((df.Open - (u.ts_sum(df.vwap, 10) / 10))) * (-1 * abs(u.rank((df.Close - df.vwap)))))


def alpha6(df):
    """
    Alpha#6
    (-1 * correlation(Open, Volume, 10))
    """
    return (-1 * u.corr(df.Open, df.Volume, 10))


def alpha7(df):
    """
    Alpha#7
    ((adv20 < Volume) ? ((-1 * ts_rank(abs(delta(Close, 7)), 60)) * sign(delta(Close, 7))) :
    (-1 * 1))
    """
    iftrue = ((-1 * u.ts_rank(abs(u.delta(df.Close, 7)), 60)) * np.sign(u.delta(df.Close, 7)))
    return pd.Series(np.where(u.adv(df, 20) < df.Volume, iftrue, (-1 * 1)), index=df.index)


def alpha8(df):
    """
    Alpha#8
    (-1 * rank(((sum(Open, 5) * sum(returns, 5)) - delay((sum(Open, 5) *
    sum(returns, 5)), 10))))
    """
    temp1 = (u.ts_sum(df.Open, 5) * u.ts_sum(df.returns, 5))
    temp2 = u.delay((u.ts_sum(df.Open, 5) * u.ts_sum(df.returns, 5)), 10)
    return (-1 * u.rank(temp1 - temp2))


def alpha9(df):
    """
    Alpha#9
    ((0 < ts_min(delta(Close, 1), 5)) ? delta(Close, 1) :
    ((ts_max(delta(Close, 1), 5) < 0) ? delta(Close, 1) : (-1 * delta(Close, 1))))
    """
    tempd1 = u.delta(df.Close, 1)
    tempmin = u.ts_min(tempd1, 5)
    tempmax = u.ts_max(tempd1, 5)
    return pd.Series(np.where(tempmin > 0, tempd1, np.where(tempmax < 0, tempd1, (-1 * tempd1))), df.index)


def alpha10(df):
    """
    Alpha#10
    rank(((0 < ts_min(delta(Close, 1), 4)) ? delta(Close, 1) : ((ts_max(delta(Close, 1), 4) < 0)
    ? delta(Close, 1) : (-1 * delta(Close, 1)))))
    """
    tempd1 = u.delta(df.Close, 1)
    tempmin = u.ts_min(tempd1, 4)
    tempmax = u.ts_max(tempd1, 4)
    return u.rank(pd.Series(np.where(tempmin > 0, tempd1, np.where(tempmax < 0, tempd1, (-1 * tempd1))), df.index))


def alpha11(df):
    """
    Alpha#11
    ((rank(ts_max((vwap - Close), 3)) +
    rank(ts_min((vwap - Close), 3))) * rank(delta(Volume, 3)))
    """
    temp1 = u.rank(u.ts_max((df.vwap - df.Close), 3))
    temp2 = u.rank(u.ts_min((df.vwap - df.Close), 3))
    temp3 = u.rank(u.delta(df.Volume, 3))
    return temp1 + (temp2 * temp3)


def alpha12(df):
    """
    Alpha#12
    (sign(delta(Volume, 1)) * (-1 * delta(Close, 1)))
    """
    return (np.sign(u.delta(df.Volume, 1)) * (-1 * u.delta(df.Close, 1)))


def alpha13(df):
    """
    Alpha#13
    (-1 * rank(covariance(rank(Close), rank(Volume), 5)))
    """
    return (-1 * u.rank(u.cov(u.rank(df.Close), u.rank(df.Volume), 5)))


def alpha14(df):
    """
    Alpha#14
    ((-1 * rank(delta(returns, 3))) * correlation(Open, Volume, 10))
    """
    return ((-1 * u.rank(u.delta(df.returns, 3))) * u.corr(df.Open, df.Volume, 10))


def alpha15(df):
    """
    Alpha#15
    (-1 * sum(rank(correlation(rank(High), rank(Volume), 3)), 3))
    """
    return (-1 * u.ts_sum(u.corr(u.rank(df.High), u.rank(df.Volume), 3), 3))


def alpha16(df):
    """
    Alpha#16
    (-1 * rank(covariance(rank(High), rank(Volume), 5)))
    """
    return (-1 * u.rank(u.cov(u.rank(df.High), u.rank(df.Volume), 5)))


def alpha17(df):
    """
    Alpha#17
    (((-1 * rank(ts_rank(Close, 10))) * rank(delta(delta(Close, 1), 1))) *
    rank(ts_rank((Volume / adv20), 5)))
    """
    temp1 = (-1 * u.rank(u.ts_rank(df.Close, 10)))
    temp2 = u.rank(u.delta(u.delta(df.Close, 1), 1))
    temp3 = u.rank(u.ts_rank((df.Volume / u.adv(df, 20)), 5))
    return ((temp1 * temp2) * temp3)


def alpha18(df):
    """
    Alpha#18
    (-1 * rank(((stddev(abs((Close - Open)), 5) + (Close - Open)) +
    correlation(Close, Open, 10))))
    """
    temp1 = u.stddev(abs((df.Close - df.Open)), 5)
    temp2 = df.Close - df.Open
    temp3 = u.corr(df.Close, df.Open, 10)
    return (-1 * u.rank(temp1 + temp2 + temp3))


def alpha19(df):
    """
    Alpha#19
    ((-1 * sign(((Close - delay(Close, 7)) + delta(Close, 7)))) *
    (1 + rank((1 + sum(returns, 250)))))
    """
    temp1 = (-1 * np.sign(((df.Close - u.delay(df.Close, 7)) + u.delta(df.Close, 7))))
    temp2 = (1 + u.rank((1 + u.ts_sum(df.returns, 250))))
    return (temp1 * temp2)


def alpha20(df):
    """
    Alpha#20
    (((-1 * rank((Open - delay(High, 1)))) * rank((Open - delay(Close, 1)))) *
    rank((Open - delay(Low, 1))))
    """
    temp1 = (-1 * u.rank((df.Open - u.delay(df.High, 1))))
    temp2 = u.rank((df.Open - u.delay(df.Close, 1)))
    temp3 = u.rank((df.Open - u.delay(df.Low, 1)))
    return (temp1 * temp2 * temp3)


def alpha21(df):
    """
    Alpha#21
    ((((sum(Close, 8) / 8) + stddev(Close, 8)) < (sum(Close, 2) / 2)) ? (-1 * 1) :
    (((sum(Close,2) / 2) < ((sum(Close, 8) / 8) - stddev(Close, 8))) ?
    1 : (((1 < (Volume / adv20)) || ((Volume /adv20) == 1)) ? 1 : (-1 * 1))))
    """
    decision1 = (u.ts_sum(df.Close, 8) / 8 + u.stddev(df.Close, 8)) < (u.ts_sum(df.Close, 2) / 2)
    decision2 = (u.ts_sum(df.Close, 2) / 2 < (u.ts_sum(df.Close, 8) / 8) - u.stddev(df.Close, 8))
    decision3 = ((1 < (df.Volume / u.adv(df, 20))) | ((df.Volume / u.adv(df, 20)) == 1))
    return np.where(decision1, (-1 * 1), np.where(decision2, 1, np.where(decision3, 1, (-1 * 1))))


def alpha22(df):
    """
    Alpha#22
    (-1 * (delta(correlation(High, Volume, 5), 5) * rank(stddev(Close, 20))))
    """
    return (-1 * (u.delta(u.corr(df.High, df.Volume, 5), 5) * u.rank(u.stddev(df.Close, 20))))


def alpha23(df):
    """
    Alpha#23
    (((sum(High, 20) / 20) < High) ? (-1 * delta(High, 2)) : 0)
    """
    return pd.Series(np.where((u.ts_sum(df.High, 20) / 20) < df.High, (-1 * u.delta(df.High, 2)), 0), df.index)


def alpha24(df):
    """
    Alpha#24
    Can be shortened without the || (or operator) and just use the <= statement.
    ((((delta((sum(Close, 100) / 100), 100) / delay(Close, 100)) < 0.05) ||
    ((delta((sum(Close, 100) / 100), 100) / delay(Close, 100)) == 0.05)) ?
    (-1 * (Close - ts_min(Close, 100))) : (-1 * delta(Close, 3)))
    """
    decision = u.delta((u.ts_sum(df.Close, 100) / 100), 100) / u.delay(df.Close, 100) <= 0.05
    if_true = (-1 * (df.Close - u.ts_min(df.Close, 100)))
    if_false = (-1 * u.delta(df.Close, 3))
    return pd.Series(np.where(decision, if_true, if_false), df.index)


def alpha25(df):
    """
    Alpha#25
    rank(((((-1 * returns) * adv20) * vwap) * (High - Close)))
    """
    return u.rank(((((-1 * df.returns) * u.adv(df, 20)) * df.vwap) * (df.High - df.Close)))


def alpha26(df):
    """
    Alpha#26
    (-1 * ts_max(correlation(ts_rank(Volume, 5), ts_rank(High, 5), 5), 3))
    """
    return (-1 * u.ts_max(u.corr(u.ts_rank(df.Volume, 5), u.ts_rank(df.High, 5), 5), 3))


def alpha27(df):
    """
    Alpha#27
    ((0.5 < rank((sum(correlation(rank(Volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
    """
    temp = np.where((0.5 < u.rank((u.ts_sum(u.corr(u.rank(df.Volume), u.rank(df.vwap), 6), 2) / 2.0))), (-1 * 1), 1)
    return pd.Series(temp, index=df.index)


def alpha28(df):
    """
    Alpha#28
    scale(((correlation(adv20, Low, 5) + ((High + Low) / 2)) - Close))
    """
    return u.scale(((u.corr(u.adv(df, 20), df.Low, 5) + ((df.High + df.Low) / 2)) - df.Close))


def alpha29(df):
    """
    Alpha#29
    (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((Close - 1),
    5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
    """
    temp1 = u.scale(np.log(u.ts_sum(u.ts_min(u.rank(u.rank((-1 * u.rank(u.delta((df.Close - 1), 5))))), 2), 1)))
    temp2 = u.product(u.rank(u.rank(temp1)), 1)
    temp3 = u.ts_rank(u.delay((-1 * df.returns), 6), 5)
    return (np.where(temp1 < temp2, temp1, temp2) + temp3)


def alpha30(df):
    """
    Alpha#30
     (((1.0 - rank(((sign((Close - delay(Close, 1))) +
     sign((delay(Close, 1) - delay(Close, 2)))) +
     sign((delay(Close, 2) - delay(Close, 3)))))) * sum(Volume, 5)) / sum(Volume, 20))
    """
    return (((1.0 - u.rank(((np.sign((df.Close - u.delay(df.Close, 1))) \
                             + np.sign((u.delay(df.Close, 1) - u.delay(df.Close, 2)))) \
                            + np.sign((u.delay(df.Close, 2) - u.delay(df.Close, 3)))))) \
             * u.ts_sum(df.Volume, 5)) / u.ts_sum(df.Volume, 20))


def alpha31(df):
    """
    Alpha#31
    ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(Close, 10)))), 10))))
    + rank((-1 * delta(Close, 3)))) + sign(scale(correlation(adv20, Low, 12))))
    """
    temp1 = u.rank(u.rank(u.rank(u.decay_linear((-1 * u.rank(u.rank(u.delta(df.Close, 10)))), 10))))
    temp2 = u.rank((-1 * u.delta(df.Close, 3))) + np.sign(u.scale(u.corr(u.adv(df, 20), df.Low, 12)))
    return temp1 + temp2


def alpha32(df):
    """
    Alpha#32
    (scale(((sum(Close, 7) / 7) - Close)) +
    (20 * scale(correlation(vwap, delay(Close, 5), 230))))
    """
    temp1 = u.scale(((u.ts_sum(df.Close, 7) / 7) - df.Close))
    temp2 = (20 * u.scale(u.corr(df.vwap, u.delay(df.Close, 5), 230)))
    return temp1 + temp2


def alpha33(df):
    """
    Alpha#33
    rank((-1 * ((1 - (Open / Close))^1)))
    """
    return u.rank((-1 * ((1 - (df.Open / df.Close)) ** 1)))


def alpha34(df):
    """
    Alpha#34
    rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(Close, 1)))))
    """
    return u.rank(((1 - u.rank((u.stddev(df.returns, 2) / u.stddev(df.returns, 5)))) \
                   + (1 - u.rank(u.delta(df.Close, 1)))))


def alpha35(df):
    """
    Alpha#35
    ((Ts_Rank(Volume, 32) * (1 - Ts_Rank(((Close + High) - Low), 16))) *
    (1 - Ts_Rank(returns, 32)))
    """
    return ((u.ts_rank(df.Volume, 32) * (1 - u.ts_rank(((df.Close + df.High) - df.Low), 16))) \
            * (1 - u.ts_rank(df.returns, 32)))


def alpha36(df):
    """
    Alpha#36
    (((((2.21 * rank(correlation((Close - Open), delay(Volume, 1), 15))) +
    (0.7 * rank((Open - Close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) +
    rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(Close, 200) / 200) - Open) * (Close - Open)))))
    """
    temp1 = (2.21 * u.rank(u.corr((df.Close - df.Open), u.delay(df.Volume, 1), 15)))
    temp2 = (0.7 * u.rank((df.Open - df.Close)))
    temp3 = (0.73 * u.rank(u.ts_rank(u.delay((-1 * df.returns), 6), 5)))
    temp4 = u.rank(abs(u.corr(df.vwap, u.adv(df, 20), 6)))
    temp5 = (0.6 * u.rank((((sum(df.Close, 200) / 200) - df.Open) * (df.Close - df.Open))))
    return ((((temp1 + temp2) + temp3) + temp4) + temp5)


def alpha37(df):
    """
    Alpha#37
    (rank(correlation(delay((Open - Close), 1), Close, 200)) + rank((Open - Close)))
    """
    return (u.rank(u.corr(u.delay((df.Open - df.Close), 1), df.Close, 200)) + u.rank((df.Open - df.Close)))


def alpha38(df):
    """
    Alpha#38
    ((-1 * rank(Ts_Rank(Close, 10))) * rank((Close / Open)))
    """
    return ((-1 * u.rank(u.ts_rank(df.Close, 10))) * u.rank((df.Close / df.Open)))


def alpha39(df):
    """
    Alpha#39

    """
    temp = (-1 * u.rank((u.delta(df.Close, 7) * (1 - u.rank(u.decay_linear((df.Volume / u.adv(df, 20)), 9))))))
    return (temp * (1 + u.rank(u.ts_sum(df.returns, 250))))


def alpha40(df):
    """
    Alpha#40
    ((-1 * rank(stddev(High, 10))) * correlation(High, Volume, 10))
    """
    return ((-1 * u.rank(u.stddev(df.High, 10))) * u.corr(df.High, df.Volume, 10))


def alpha41(df):
    """
    Alpha#41
    (((High * Low)^0.5) - vwap)
    """
    return (((df.High * df.Low) ** 0.5) - df.vwap)


def alpha42(df):
    """
    Alpha#42
    (rank((vwap - Close)) / rank((vwap + Close)))
    """
    return (u.rank((df.vwap - df.Close)) / u.rank((df.vwap + df.Close)))


def alpha43(df):
    """
    Alpha#43
    (ts_rank((Volume / adv20), 20) * ts_rank((-1 * delta(Close, 7)), 8))
    """
    return (u.ts_rank((df.Volume / u.adv(df, 20)), 20) * u.ts_rank((-1 * u.delta(df.Close, 7)), 8))


def alpha44(df):
    """
    Alpha#44
    (-1 * correlation(High, rank(Volume), 5))
    """
    return (-1 * u.corr(df.High, u.rank(df.Volume), 5))


def alpha45(df):
    """
    Alpha#45
    (-1 * ((rank((sum(delay(Close, 5), 20) / 20)) * correlation(Close, Volume, 2))
    * rank(correlation(sum(Close, 5), sum(Close, 20), 2))))
    """
    temp1 = u.rank((u.ts_sum(u.delay(df.Close, 5), 20) / 20))
    temp2 = u.corr(df.Close, df.Volume, 2)
    temp3 = u.rank(u.corr(u.ts_sum(df.Close, 5), u.ts_sum(df.Close, 20), 2))
    return (-1 * ((temp1 * temp2) * temp3))


def alpha46(df):
    """
    Alpha#46
    ((0.25 < (((delay(Close, 20) - delay(Close, 10)) / 10) - ((delay(Close, 10) - Close) / 10))) ? (-1 * 1) :
    (((((delay(Close, 20) - delay(Close, 10)) / 10) - ((delay(Close, 10) - Close) / 10)) < 0) ? 1 :
    ((-1 * 1) * (Close - delay(Close, 1)))))
    """
    decision1 = (0.25 < (
                ((u.delay(df.Close, 20) - u.delay(df.Close, 10)) / 10) - ((u.delay(df.Close, 10) - df.Close) / 10)))
    decision2 = ((((u.delay(df.Close, 20) - u.delay(df.Close, 10)) / 10) - (
                (u.delay(df.Close, 10) - df.Close) / 10)) < 0)
    iffalse = ((-1 * 1) * (df.Close - u.delay(df.Close, 1)))
    return pd.Series(np.where(decision1, (-1 * 1), np.where(decision2, 1, iffalse)), index=df.index)


def alpha47(df):
    """
    Alpha#47
    ((((rank((1 / Close)) * Volume) / adv20) *
    ((High * rank((High - Close))) / (sum(High, 5) / 5))) - rank((vwap - delay(vwap, 5))))
    """
    temp1 = ((u.rank((1 / df.Close)) * df.Volume) / u.adv(df, 20))
    temp2 = ((df.High * u.rank((df.High - df.Close))) / (u.ts_sum(df.High, 5) / 5))
    return ((temp1 * temp2) - u.rank((df.vwap - u.delay(df.vwap, 5))))


def alpha49(df):
    """
    Alpha#49
    (((((delay(Close, 20) - delay(Close, 10)) / 10) - ((delay(Close, 10) - Close) / 10)) <
    (-1 * 0.1)) ? 1 : ((-1 * 1) * (Close - delay(Close, 1))))
    """
    temp1 = ((u.delay(df.Close, 20) - u.delay(df.Close, 10)) / 10)
    temp2 = ((u.delay(df.Close, 10) - df.Close) / 10)
    return pd.Series(np.where(((temp1 - temp2) < (-1 * 0.1)), 1, ((-1 * 1) * (df.Close - u.delay(df.Close, 1)))),
                     index=df.index)


def alpha50(df):
    """
    Alpha#50
    (-1 * ts_max(rank(correlation(rank(Volume), rank(vwap), 5)), 5))
    """
    return (-1 * u.ts_max(u.rank(u.corr(u.rank(df.Volume), u.rank(df.vwap), 5)), 5))


def alpha51(df):
    """
    Alpha#51
    (((((delay(Close, 20) - delay(Close, 10)) / 10) - ((delay(Close, 10) - Close) / 10))
    < (-1 * 0.05)) ? 1 : ((-1 * 1) * (Close - delay(Close, 1))))
    """
    condition = ((((u.delay(df.Close, 20) - u.delay(df.Close, 10)) / 10) \
                  - ((u.delay(df.Close, 10) - df.Close) / 10)) < (-1 * 0.05))
    return pd.Series(np.where(condition, 1, ((-1 * 1) * (df.Close - u.delay(df.Close, 1)))), df.index)


def alpha52(df):
    """
    Alpha#52
    ((((-1 * ts_min(Low, 5)) + delay(ts_min(Low, 5), 5)) * rank(((sum(returns, 240)
    - sum(returns, 20)) / 220))) * ts_rank(Volume, 5))
    """
    temp1 = ((-1 * u.ts_min(df.Low, 5)) + u.delay(u.ts_min(df.Low, 5), 5))
    temp2 = u.rank(((u.ts_sum(df.returns, 240) - u.ts_sum(df.returns, 20)) / 220))
    return ((temp1 * temp2) * u.ts_rank(df.Volume, 5))


def alpha53(df):
    """
    Alpha#53
    (-1 * delta((((Close - Low) - (High - Close)) / (Close - Low)), 9))
    """
    return (-1 * u.delta((((df.Close - df.Low) - (df.High - df.Close)) / (df.Close - df.Low)), 9))


def alpha54(df):
    """
    Alpha#54
    ((-1 * ((Low - Close) * (Open^5))) / ((Low - High) * (Close^5)))
    """
    return ((-1 * ((df.Low - df.Close) * (df.Open ** 5))) / ((df.Low - df.High) * (df.Close ** 5)))


def alpha55(df):
    """
    Alpha#55
    (-1 * correlation(rank(((Close - ts_min(Low, 12)) / (ts_max(High, 12)
    - ts_min(Low, 12)))), rank(Volume), 6))
    """
    temp1 = (df.Close - u.ts_min(df.Low, 12))
    temp2 = (u.ts_max(df.High, 12) - u.ts_min(df.Low, 12))
    return (-1 * u.corr(u.rank((temp1 / temp2)), u.rank(df.Volume), 6))



def alpha57(df):
    """
    Alpha#57
    (0 - (1 * ((Close - vwap) / decay_linear(rank(ts_argmax(Close, 30)), 2))))
    """
    return (0 - (1 * ((df.Close - df.vwap) / u.decay_linear(u.rank(u.ts_argmax(df.Close, 30)), 2))))


def alpha60(df):
    """
    Alpha#60
    (0 - (1 * ((2 * scale(rank(((((Close - Low) - (High - Close)) / (High - Low)) * Volume))))
    - scale(rank(ts_argmax(Close, 10))))))
    """
    temp1 = u.scale(u.rank(((((df.Close - df.Low) - (df.High - df.Close)) / (df.High - df.Low)) * df.Volume)))
    return (0 - (1 * ((2 * temp1) - u.scale(u.rank(u.ts_argmax(df.Close, 10))))))


def alpha61(df):
    """
    Alpha#61
    (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))

    Rounded the days to int since partial lookback
    """
    return (u.rank((df.vwap - u.ts_min(df.vwap, 16))) < u.rank(u.corr(df.vwap, u.adv(df, 180), 18)))


def alpha62(df):
    """
    Alpha#62
    ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(Open)
    + rank(Open)) < (rank(((High + Low) / 2)) + rank(High))))) * -1)
    """
    temp1 = u.rank(u.corr(df.vwap, u.ts_sum(u.adv(df, 20), 22), 10))
    temp2 = u.rank(((u.rank(df.Open) + u.rank(df.Open)) < (u.rank(((df.High + df.Low) / 2)) + u.rank(df.High))))
    return ((temp1 < temp2) * -1)


def alpha64(df):
    """
    Alpha#64
    ((rank(correlation(sum(((Open * 0.178404) + (Low * (1 - 0.178404))), 12.7054),
    sum(adv120, 12.7054), 16.6208)) < rank(delta(((((High + Low) / 2) * 0.178404)
    + (vwap * (1 -0.178404))), 3.69741))) * -1)
    """
    temp1 = u.ts_sum(((df.Open * 0.178404) + (df.Low * (1 - 0.178404))), 13)
    temp2 = u.rank(u.corr(temp1, u.ts_sum(u.adv(df, 120), 18), 17))
    temp3 = u.rank(u.delta(((((df.High + df.Low) / 2) * 0.178404) + (df.vwap * (1 - 0.178404))), 4))
    return ((temp2 < temp3) * -1)


def alpha65(df):
    """
    Alpha#65
    ((rank(correlation(((Open * 0.00817205) +
    (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((Open - ts_min(Open, 13.635)))) * -1)
    """
    temp1 = (df.Open * 0.00817205) + (df.vwap * (1 - 0.00817205))
    temp2 = u.rank((df.Open - u.ts_min(df.Open, 14)))
    return ((u.rank(u.corr(temp1, u.ts_sum(u.adv(df, 60), 9), 6)) < temp2) * -1)


def alpha66(df):
    """
    Alpha#66
    ((rank(decay_linear(delta(vwap, 3.51013), 7.23052))
    + Ts_Rank(decay_linear(((((Low * 0.96633)
    + (Low * (1 - 0.96633))) - vwap) / (Open - ((High + Low) / 2))), 11.4157), 6.72611)) * -1)
    """
    temp1 = u.rank(u.decay_linear(u.delta(df.vwap, 4), 7.23052))
    temp2 = (((df.Low * 0.96633) + (df.Low * (1 - 0.96633))) - df.vwap)
    temp3 = (df.Open - ((df.High + df.Low) / 2))
    return ((temp1 + u.ts_rank(u.decay_linear((temp2 / temp3), 11.4157), 7)) * -1)

def alpha68(df):
    """
    Alpha#68
    ((Ts_Rank(correlation(rank(High), rank(adv15), 8.91644), 13.9333)
    < rank(delta(((Close * 0.518371) + (Low * (1 - 0.518371))), 1.06157))) * -1)
    """
    temp1 = u.ts_rank(u.corr(u.rank(df.High), u.rank(u.adv(df, 15)), 9), 14)
    temp2 = u.rank(u.delta(((df.Close * 0.518371) + (df.Low * (1 - 0.518371))), 1))
    return u.rank(u.delta(((df.Close * 0.518371) + (df.Low * (1 - 0.518371))), 1))



def alpha71(df):
    """
    Alpha#71
    max(Ts_Rank(decay_linear(correlation(Ts_Rank(Close, 3.43976),
    Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((Low + Open)
    - (vwap + vwap)))^2), 16.4662), 4.4388))
    """
    temp1 = u.corr(u.ts_rank(df.Close, 3), u.ts_rank(u.adv(df, 180), 12), 18)
    temp2 = u.ts_rank(u.decay_linear((u.rank(((df.Low + df.Open) - (df.vwap + df.vwap))) ** 2), 16.4662), 4)
    return pd.Series(np.where(temp1 > temp2, temp1, temp2), df.index)


def alpha72(df):
    """
    Alpha#72
    (rank(decay_linear(correlation(((High + Low) / 2), adv40, 8.93345), 10.1519)) /
    rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469),
    Ts_Rank(Volume, 18.5188), 6.86671),2.95011)))
    """
    temp1 = u.rank(u.decay_linear(u.corr(((df.High + df.Low) / 2), u.adv(df, 40), 9), 10))
    temp2 = u.rank(u.decay_linear(u.corr(u.ts_rank(df.vwap, 4), u.ts_rank(df.Volume, 19), 7), 2.95011))
    return (temp1 / temp2)


def alpha73(df):
    """
    Alpha#73
    (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),
    Ts_Rank(decay_linear(((delta(((Open * 0.147155) + (Low * (1 - 0.147155))), 2.03608)
    / ((Open * 0.147155) + (Low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
    """
    temp1 = u.rank(u.decay_linear(u.delta(df.vwap, 5), 2.91864))
    temp2 = u.delta(((df.Open * 0.147155) + (df.Low * (1 - 0.147155))), 2)
    temp3 = ((df.Open * 0.147155) + (df.Low * (1 - 0.147155)))
    temp4 = u.ts_rank(u.decay_linear(((temp2 / temp3) * -1), 2), 17)
    return pd.Series(np.where(temp1 > temp4, temp1 * -1, temp4 * -1), df.index)


def alpha74(df):
    """
    Alpha#74
    ((rank(correlation(Close, sum(adv30, 37.4843), 15.1365))
    < rank(correlation(rank(((High * 0.0261661)
    + (vwap * (1 - 0.0261661)))), rank(Volume), 11.4791))) * -1)
    """
    temp1 = u.rank(u.corr(df.Close, u.ts_sum(u.adv(df, 30), 37), 15))
    temp2 = u.rank(u.corr(u.rank(((df.High * 0.0261661) + (df.vwap * (1 - 0.0261661)))), u.rank(df.Volume), 11))
    return ((temp1 < temp2) * -1)


def alpha75(df):
    """
    Alpha#75(df)
    (rank(correlation(vwap, Volume, 4.24304))
    < rank(correlation(rank(Low), rank(adv50), 12.4413)))
    """
    temp1 = u.rank(u.corr(df.vwap, df.Volume, 4))
    temp2 = u.rank(u.corr(u.rank(df.Low), u.rank(u.adv(df, 50)), 12))
    return (temp1 < temp2)



def alpha77(df):
    """
    Alpha#77
    min(rank(decay_linear(((((High + Low) / 2) + High) - (vwap + High)), 20.0451)),
    rank(decay_linear(correlation(((High + Low) / 2), adv40, 3.1614), 5.64125)))
    """
    temp1 = u.rank(u.decay_linear(((((df.High + df.Low) / 2) + df.High) - (df.vwap + df.High)), 20.0451))
    temp2 = u.rank(u.decay_linear(u.corr(((df.High + df.Low) / 2), u.adv(df, 40), 3), 5.64125))
    return pd.Series(np.where(temp1 > temp2, temp1, temp2), index=df.index)


def alpha78(df):
    """
    Alpha#78
    (rank(correlation(sum(((Low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),
    sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(Volume), 5.77492)))
    """
    temp1 = u.ts_sum(((df.Low * 0.352233) + (df.vwap * (1 - 0.352233))), 20)
    temp2 = u.rank(u.corr(u.rank(df.vwap), u.rank(df.Volume), 6))
    temp3 = u.rank(u.corr(temp1, u.ts_sum(u.adv(df, 40), 20), 7))
    return (temp3 ** temp2)



def alpha81(df):
    """
    Alpha#81
    ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054),
    8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(Volume), 5.07914))) * -1)
    """
    temp = u.rank(np.log(u.product(u.rank(u.rank(u.corr(df.vwap, u.ts_sum(u.adv(df, 10), 50), 8)) ** 4), 15)))
    return ((temp < u.rank(u.corr(u.rank(df.vwap), u.rank(df.Volume), 5))) * -1)


def alpha83(df):
    """
    Alpha#83
    ((rank(delay(((High - Low) / (sum(Close, 5) / 5)), 2)) * rank(rank(Volume)))
    / (((High - Low) / (sum(Close, 5) / 5)) / (vwap - Close)))
    """
    temp1 = u.rank(u.delay(((df.High - df.Low) / (u.ts_sum(df.Close, 5) / 5)), 2)) * u.rank(u.rank(df.Volume))
    temp2 = (((df.High - df.Low) / (u.ts_sum(df.Close, 5) / 5)) / (df.vwap - df.Close))
    return (temp1 / temp2)


def alpha84(df):
    """
    Alpha#84
    SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127),
    delta(Close, 4.96796))
    """
    return (u.ts_rank((df.vwap - u.ts_max(df.vwap, 15)), 21) ** u.delta(df.Close, 5))


def alpha85(df):
    """
    Alpha#85
    (rank(correlation(((High * 0.876703) + (Close * (1 - 0.876703))), adv30,
    9.61331))^rank(correlation(Ts_Rank(((High + Low) / 2), 3.70596),
    Ts_Rank(Volume, 10.1595), 7.11408)))
    """
    temp1 = u.rank(u.corr(((df.High * 0.876703) + (df.Close * (1 - 0.876703))), u.adv(df, 30), 10))
    temp2 = u.rank(u.corr(u.ts_rank(((df.High + df.Low) / 2), 4), u.ts_rank(df.Volume, 10), 7))
    return (temp1 ** temp2)


def alpha86(df):
    """
    Alpha#86
    ((Ts_Rank(correlation(Close, sum(adv20, 14.7444), 6.00049), 20.4195)
    < rank(((Open + Close) - (vwap + Open)))) * -1)
    """
    temp1 = u.ts_rank(u.corr(df.Close, u.ts_sum(u.adv(df, 20), 15), 6), 20)
    temp2 = u.rank(((df.Open + df.Close) - (df.vwap + df.Open)))
    return ((temp1 < temp2) * -1)



def alpha88(df):
    """
    Alpha#88
    min(rank(decay_linear(((rank(Open) + rank(Low)) - (rank(High) + rank(Close))),
    8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(Close, 8.44728), Ts_Rank(adv60,
    20.6966), 8.01266), 6.65053), 2.61957))
    """
    temp1 = u.rank(u.decay_linear(((u.rank(df.Open) + u.rank(df.Low)) - (u.rank(df.High) + u.rank(df.Close))), 8))
    temp2 = u.ts_rank(u.decay_linear(u.corr(u.ts_rank(df.Close, 8), u.ts_rank(u.adv(df, 60), 21), 8), 6.65053), 3)
    return pd.Series(np.where(temp1 < temp2, temp1, temp2), index=df.index)




def alpha92(df):
    """
    Alpha#92
    min(Ts_Rank(decay_linear(((((High + Low) / 2) + Close) < (Low + Open)), 14.7221),
    18.8683), Ts_Rank(decay_linear(correlation(rank(Low), rank(adv30), 7.58555), 6.94024),
    6.80584))
    """
    temp1 = u.ts_rank(u.decay_linear(((((df.High + df.Low) / 2) + df.Close) < (df.Low + df.Open)), 14.7221), 19)
    temp2 = u.ts_rank(u.decay_linear(u.corr(u.rank(df.Low), u.rank(u.adv(df, 30)), 8), 6.94024), 7)
    return pd.Series(np.where(temp1 < temp2, temp1, temp2), index=df.index)



def alpha94(df):
    """
    Alpha#94
    ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap,
    19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    """
    temp1 = u.rank((df.vwap - u.ts_min(df.vwap, 12)))
    temp2 = u.ts_rank(u.corr(u.ts_rank(df.vwap, 20), u.ts_rank(u.adv(df, 60), 4), 18), 3)
    return ((temp1 ** temp2) * -1)


def alpha95(df):
    """
    Alpha#95
    (rank((Open - ts_min(Open, 12.4105))) < Ts_Rank((rank(correlation(sum(((High + Low)
    / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
    """
    temp1 = u.rank((df.Open - u.ts_min(df.Open, 12)))
    temp2 = u.corr(u.ts_sum(((df.High + df.Low) / 2), 19), u.ts_sum(u.adv(df, 40), 19), 13)
    return (temp1 < u.ts_rank((u.rank(temp2) ** 5), 12))


def alpha96(df):
    """
    Alpha#96
    (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(Volume), 3.83878),
    4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(Close, 7.45404),
    Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
    """
    temp1 = u.ts_rank(u.decay_linear(u.corr(u.rank(df.vwap), u.rank(df.Volume), 4), 4.16783), 8)
    temp2 = u.corr(u.ts_rank(df.Close, 7), u.ts_rank(u.adv(df, 60), 4), 4)
    temp3 = u.ts_rank(u.decay_linear(u.ts_argmax(temp2, 13), 14.0365), 13)
    return pd.Series(np.where(temp1 > temp3, temp1, temp3), index=df.index)



def alpha98(df):
    """
    Alpha#98
    (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) -
    rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(Open), rank(adv15), 20.8187), 8.62571),
    6.95668), 8.07206)))
    """
    temp1 = u.ts_rank(u.ts_argmin(u.corr(u.rank(df.Open), u.rank(u.adv(df, 15)), 21), 9), 7)
    temp2 = u.rank(u.decay_linear(temp1, 8.07206))
    temp3 = u.rank(u.decay_linear(u.corr(df.vwap, u.ts_sum(u.adv(df, 5), 26), 5), 7))
    return (temp3 - temp2)


def alpha99(df):
    """
    Alpha#99
    ((rank(correlation(sum(((High + Low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136))
    < rank(correlation(Low, Volume, 6.28259))) * -1)
    """
    temp1 = u.rank(u.corr(u.ts_sum(((df.High + df.Low) / 2), 20), u.ts_sum(u.adv(df, 60), 20), 9))
    temp2 = u.rank(u.corr(df.Low, df.Volume, 6))
    return pd.Series(np.where(temp1 < temp2, temp1 * -1, temp2 * -1), index=df.index)



def alpha101(df):
    """
    Alpha#101
    ((Close - Open) / ((High - Low) + .001))
    """
    return ((df.Close - df.Open) / ((df.High - df.Low) + .001))


########################ALFAS NO DISPONIBLES###################################
def alpha48(df):
    """
    Alpha#48
    (indneutralize(((correlation(delta(Close, 1), delta(delay(Close, 1), 1), 250) *
    delta(Close, 1)) / Close), IndClass.subindustry) / sum(((delta(Close, 1) / delay(Close, 1))^2), 250))
    """
    pass

def alpha56(df):
    """
    Alpha#56
    (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
    No MarketCap Available
    """
    pass

def alpha58(df):
    """
    Alpha#58
    (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector),
    Volume, 3.92795), 7.89291), 5.50322))
    """
    pass


def alpha59(df):
    """
    Alpha#59
    (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap *
    (1 - 0.728317))), IndClass.industry), Volume, 4.25197), 16.2289), 8.19648))
    """
    pass

def alpha63(df):
    """
    Alpha#63
    ((rank(decay_linear(delta(IndNeutralize(Close, IndClass.industry), 2.25164), 8.22237))
    - rank(decay_linear(correlation(((vwap * 0.318108) + (Open * (1 - 0.318108))),
    sum(adv180, 37.2467), 13.557), 12.2883))) * -1)
    """
    pass

def alpha67(df):
    """
    Alpha#67
    ((rank((High - ts_min(High, 2.14593)))^rank(correlation(IndNeutralize(vwap,
    IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
    """
    pass

def alpha69(df):
    """
    Alpha#69
    ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412),
    4.79344))^Ts_Rank(correlation(((Close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416),
    9.0615)) * -1)
    """
    pass


def alpha70(df):
    """
    Alpha#70
    ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(Close,
    IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
    """
    pass

def alpha76(df):
    """
    Alpha#76
    (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),
    Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(Low, IndClass.sector), adv81,
    8.14941), 19.569), 17.1543), 19.383)) * -1)
    """
    pass

def alpha79(df):
    """
    Alpha#79
    (rank(delta(IndNeutralize(((Close * 0.60733) + (Open * (1 - 0.60733))),
    IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973),
    Ts_Rank(adv150, 9.18637), 14.6644)))
    """
    pass


def alpha80(df):
    """
    Alpha#80
    ((rank(Sign(delta(IndNeutralize(((Open * 0.868128) + (High * (1 - 0.868128))),
    IndClass.industry), 4.04545)))^Ts_Rank(correlation(High, adv10, 5.11456), 5.53756)) * -1)
    """
    pass

def alpha82(df):
    """
    Alpha#82
    (min(rank(decay_linear(delta(Open, 1.46063), 14.8717)),
    Ts_Rank(decay_linear(correlation(IndNeutralize(Volume, IndClass.sector),
    ((Open * 0.634196) + (Open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
    """
    pass

def alpha87(df):
    """
    Alpha#87
    (max(rank(decay_linear(delta(((Close * 0.369701) + (vwap * (1 - 0.369701))),
    1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81,
    IndClass.industry), Close, 13.4132)), 4.89768), 14.4535)) * -1)
    """
    pass

def alpha89(df):
    """
    Alpha#89
    (Ts_Rank(decay_linear(correlation(((Low * 0.967285) + (Low * (1 - 0.967285))), adv10,
    6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap,
    IndClass.industry), 3.48158), 10.1466), 15.3012))
    """
    pass


def alpha90(df):
    """
    Alpha#90
    ((rank((Close - ts_max(Close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40,
    IndClass.subindustry), Low, 5.38375), 3.21856)) * -1)
    """
    pass


def alpha91(df):
    """
    Alpha#91
    ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(Close,
    IndClass.industry), Volume, 9.74928), 16.398), 3.83219), 4.8667) -
    rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
    """
    pass

def alpha93(df):
    """
    Alpha#93
    (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81,
    17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((Close * 0.524434)
    + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))
    """
    pass

def alpha97(df):
    """
    Alpha#97
    ((rank(decay_linear(delta(IndNeutralize(((Low * 0.721001) + (vwap * (1 - 0.721001))),
    IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(Low,
    7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
    """
    pass

def alpha100(df):
    """
    Alpha#100
    (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((Close - Low)
    - (High - Close)) / (High - Low)) * Volume)), IndClass.subindustry), IndClass.subindustry)))
    - scale(indneutralize((correlation(Close, rank(adv20), 5) - rank(ts_argmin(Close, 30))),
    IndClass.subindustry))) * (Volume / adv20))))
    """
    pass
