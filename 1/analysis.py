import warnings
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss


def load_data(path="data/Online Retail.xlsx"):
    path = Path(path)
    if not path.exists():
        url = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
        with urlopen(url, timeout=120) as response:
            archive = ZipFile(BytesIO(response.read()))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.read("Online Retail.xlsx"))
    data = pd.read_excel(path, dtype={"InvoiceNo": str, "StockCode": str})
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="raise")
    return data


def data_summary(data):
    return pd.DataFrame({"тип": data.dtypes.astype(str), "пропуски": data.isna().sum(), "уникальные": data.nunique()})


def select_product(data, code):
    product = data.loc[data["StockCode"].eq(code)].copy()
    unique = product.drop_duplicates()
    cancelled = unique["InvoiceNo"].str.upper().str.startswith("C")
    valid = unique["Quantity"].gt(0) & unique["UnitPrice"].gt(0) & ~cancelled
    sales = unique.loc[valid].copy()
    report = pd.Series(
        {
            "Строк товара": len(product),
            "Точных повторов": int(product.duplicated().sum()),
            "Количество в повторах, шт.": product.loc[product.duplicated(), "Quantity"].sum(),
            "Отмен и возвратов, строк": int(cancelled.sum()),
            "Отмен и возвратов, шт.": -unique.loc[cancelled, "Quantity"].sum(),
            "Прочих исключённых строк": int((~valid & ~cancelled).sum()),
            "Оставлено строк": len(sales),
            "Продано за весь файл, шт.": sales["Quantity"].sum(),
        }
    )
    return product, sales, report


def make_calendar(data, sales):
    dates = pd.date_range(data["InvoiceDate"].min().normalize(), data["InvoiceDate"].max().normalize(), freq="D")
    all_rows = data.groupby(data["InvoiceDate"].dt.normalize()).size()
    units = sales.groupby(sales["InvoiceDate"].dt.normalize())["Quantity"].sum()
    calendar = pd.DataFrame(index=dates)
    calendar.index.name = "date"
    calendar["transactions"] = all_rows.reindex(dates, fill_value=0)
    calendar["units"] = units.reindex(dates)  # пока не заменяем отсутствие записей
    calendar["status"] = np.select(
        [calendar["units"].notna(), calendar["transactions"].gt(0)],
        ["Есть продажи товара", "Есть операции магазина, нет продаж товара"],
        default="Нет операций магазина",
    )
    return calendar


def aggregate_sales(daily):
    result = {}
    for name, freq in [("День", "D"), ("Неделя", "W-SUN"), ("Месяц", "MS")]:
        values = daily.resample(freq).sum(min_count=1)
        count = daily.resample(freq).count()
        expected = values.index.days_in_month if freq == "MS" else (7 if freq == "W-SUN" else 1)
        result[name] = pd.DataFrame({"units": values, "days": count, "complete": count == expected})
    return result


def iqr_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return (series < lower) | (series > upper), lower, upper


def describe_series(series):
    result = series.describe()
    result["нулей, %"] = 100 * series.eq(0).mean()
    result["CV"] = series.std() / series.mean()
    result["асимметрия"] = series.skew()
    result["ACF(1)"] = series.autocorr(1)
    return result


def plot_aggregations(aggregated):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), layout="constrained")
    for ax, (name, frame) in zip(axes, aggregated.items()):
        ax.plot(frame.index, frame["units"], label="Продажи")
        partial = frame.loc[~frame["complete"]]
        if not partial.empty:
            ax.scatter(partial.index, partial["units"], c="tab:red", zorder=3, label="Неполный период")
        ax.set(title=name, xlabel="Дата", ylabel="Продажи, шт.")
        ax.legend()
    return fig


def plot_quality(series_by_freq):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), layout="constrained")
    for ax, (name, series) in zip(axes, series_by_freq.items()):
        mask, _, upper = iqr_outliers(series)
        ax.plot(series, label="Продажи")
        if mask.any():
            ax.scatter(series[mask].index, series[mask], c="tab:red", label="За границами IQR")
        ax.axhline(upper, ls="--", c="grey", label="Верхняя граница")
        ax.set(title=f"{name}: возможные выбросы", xlabel="Дата", ylabel="Продажи, шт.")
        ax.legend()
    return fig


def plot_rolling(series_by_freq):
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), layout="constrained")
    for row, (name, series), window in zip(axes, series_by_freq.items(), [28, 4, 3]):
        rolling = series.rolling(window)
        row[0].plot(series, alpha=0.35, label="Продажи")
        row[0].plot(rolling.mean(), label=f"Среднее, окно {window}")
        row[0].set(title=f"{name}: уровень", xlabel="Дата", ylabel="Продажи, шт.")
        row[0].legend()
        row[1].plot(rolling.std())
        row[1].set(title=f"{name}: разброс, окно {window}", xlabel="Дата", ylabel="СКО, шт.")
        for ax in row:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    return fig


def decompose(series, period=None, window=5):
    if period is not None:
        result = STL(series, period=period, seasonal=13, robust=True).fit()
        return pd.DataFrame(
            {"Ряд": series, "Тренд": result.trend, "Сезонность": result.seasonal, "Остаток": result.resid}
        )
    trend = series.rolling(window, center=True, min_periods=1).mean()
    return pd.DataFrame({"Ряд": series, "Тренд": trend, "Остаток": series - trend})


def plot_components(components, title):
    fig, axes = plt.subplots(
        len(components.columns), 1, figsize=(11, 2 * len(components.columns)), sharex=True, layout="constrained"
    )
    for ax, column in zip(axes, components):
        ax.plot(components.index, components[column])
        ax.set(title=column, ylabel="Шт.")
    axes[-1].set_xlabel("Дата")
    fig.suptitle(title)
    return fig


def weekday_profile(series):
    result = series.groupby(series.index.day_of_week).agg(["mean", "median", "std"])
    result.index = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    return result


def plot_seasonality(series_by_freq):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), layout="constrained")
    for ax, (name, series), lag in zip(axes, series_by_freq.items(), [28, 20, 5]):
        plot_acf(series, lags=lag, zero=False, ax=ax, title=f"{name}: автокорреляция")
        ax.set(xlabel="Лаг, периодов", ylabel="ACF")
    return fig


def plot_weekday_and_spectrum(daily):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    weekday_profile(daily)[["mean", "median"]].rename(columns={"mean": "Среднее", "median": "Медиана"}).plot.bar(
        ax=axes[0], rot=0
    )
    axes[0].set(title="Продажи по дням недели", xlabel="День недели", ylabel="Продажи, шт.")
    frequencies, power = periodogram(daily, fs=1, detrend="linear", scaling="spectrum")
    axes[1].plot(frequencies[1:], power[1:])
    axes[1].axvline(1 / 7, ls="--", c="tab:red", label="Период 7 дней")
    axes[1].set(title="Спектр дневного ряда", xlabel="Частота, циклов/день", ylabel="Мощность, шт.²")
    axes[1].legend()
    return fig


def stationarity(series, maxlag):
    series = series.dropna()
    adf = adfuller(series, regression="c", maxlag=maxlag, autolag="AIC", result_object=False)
    # короткое правило Шверта
    nlags = int(4 * (len(series) / 100) ** 0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        test = kpss(series, regression="c", nlags=nlags, result_object=False)
    kpss_p = ">= 0.100" if test[1] >= 0.1 else "<= 0.010" if test[1] <= 0.01 else f"{test[1]:.3f}"
    return pd.Series(
        {
            "n": len(series),
            "ADF": adf[0],
            "p ADF": f"{adf[1]:.3g}",
            "лаг ADF": adf[2],
            "KPSS": test[0],
            "p KPSS": kpss_p,
            "лаги KPSS": test[2],
            "тесты согласны со стационарностью": adf[1] < 0.05 and test[1] > 0.05,
        }
    )


def difference(series, lag=1):
    return series.diff(lag).dropna()


def save_series(aggregated, transformed, path="results"):
    path = Path(path)
    path.mkdir(exist_ok=True)
    for name, filename in [("День", "daily"), ("Неделя", "weekly"), ("Месяц", "monthly")]:
        aggregated[name].to_csv(path / f"{filename}.csv", index_label="date")
    transformed.to_csv(path / "daily_diff7.csv", index_label="date")
