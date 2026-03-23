# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.0.0",
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "polars==1.39.3",
#     "scipy==1.17.1",
#     "statsmodels==0.14.6",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Time series

    **[Fernando Náufel](https://fnaufel.github.io)**

    2026-03-23
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## About this document

    This web page is a [marimo notebook](https://docs.marimo.io/), a type of Python notebook that has quite a few advantages over Jupyter notebooks. Read more about it at [the marimo website](https://docs.marimo.io/).

    The Python interpreter and the necessary packages are all running **in your browser**, using [Pyodide](https://pyodide.org/en/stable/) and [WebAssembly](https://webassembly.org). Nothing will be installed on your device.

    What you see now is a **read-only, interactive version of the notebook**. All cells should be executed when this web page is loaded, and their outputs should be rendered after a while.

    If you suspect there is something wrong, click on the '...' button on the top right and then on 'show code'. If there are any error messages, you should be able to see them.

    If you want to fork and edit this marimo notebook, [open it in molab](https://molab.marimo.io/github/fnaufel/time-series-marimo/blob/master/time-series.py).

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/fnaufel/time-series-marimo/blob/master/time-series.py)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Introduction
    """)
    return


@app.cell
def _(mo):
    mo.md(
        '*It is difficult to make predictions, especially about the future.*  '
        '\n'
        'NIELS BOHR, Danish physicist'
    ).right()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Imports
    """)
    return


@app.cell
def _():
    import marimo as mo
    mo.show_code()
    return (mo,)


@app.cell
def _(mo):
    import numpy as np
    import polars as pl
    import altair as alt
    alt.data_transformers.enable('marimo_csv')
    from scipy.stats import boxcox
    from scipy.special import inv_boxcox
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf
    import matplotlib
    mo.show_code()
    return ARIMA, alt, boxcox, inv_boxcox, np, pl, plot_acf, seasonal_decompose


@app.cell
def _(mo):
    mo.md(r"""
    ### Example: airline passenger data
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    As our main example, we will use data about the number of passengers in flights from 1949 to 1960. The file is available online [at the Kaggle site](https://www.kaggle.com/datasets/ashfakyeafi/air-passenger-data-for-time-series-analysis).
    """)
    return


@app.cell
def _(mo, pl):
    path_to_csv = 'https://raw.githubusercontent.com/fnaufel/time-series-marimo/refs/heads/master/data/AirPassengers.csv'
    df = pl.read_csv(path_to_csv).with_columns(
        Month = (pl.col("Month") + "-01").str.to_date("%Y-%m-%d")
    ).rename({'#Passengers': 'Passengers'})

    mo.show_code(df)
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    The first step should always be to **visualize the data**.

    For time series, line graphs are usually the most useful.
    """)
    return


@app.cell
def _(df, january_labeled_monthly_line_chart, mo):
    chart_original = january_labeled_monthly_line_chart(
        df,
        y="Passengers",
        title="Passengers by Month",
    )

    mo.show_code(chart_original)
    return (chart_original,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Stationarity
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    A time series is **stationary** if the mean and the standard deviation are constant through time.

    In our example, we can verify that the time series is **not** stationary. As the years go by, the mean goes up and the values are more spread out (the standard deviation goes up).

    In the chart below, select a time interval by using the mouse and check the statistics for the selected points:
    """)
    return


@app.cell
def _(chart_original, mo):
    chart_select = mo.ui.altair_chart(
        chart_original
    )

    chart_select
    return (chart_select,)


@app.cell
def _(chart_select, df, mo, selection_summary_md):
    _df_selected = chart_select.apply_selection(df)

    mo.md(
        selection_summary_md(
            _df_selected,
            value_column="Passengers",
            value_label="passengers",
        )
    ).center()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### How to make the series stationary
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Differencing to make the mean constant

    Let's write $y_t$ for the number of passengers at time $t$. Instead of working with the values
    $$
    y_t
    $$
    we work with the **discrete differences** $y'$
    $$
    y'_t = y_t - y_{t-1}
    $$

    We may difference the resulting series again, obtaining a second-order differenced series $y^*$:
    $$
    \begin{aligned}
    y_{t}^{*}
    &=y_{t}'-y_{t-1}'\\
    &=(y_{t}-y_{t-1})-(y_{t-1}-y_{t-2})\\
    &=y_{t}-2y_{t-1}+y_{t-2}
    \end{aligned}
    $$

    The `diff()` method (in Pandas or in Polars) computes the discrete difference of a series (which is also a series).

    Choose the value of the differencing order $k$ in the slider below to see how the graph changes, then select a time range in the graph to see the statistics for it.
    """)
    return


@app.cell
def _(df, diff_order, january_labeled_monthly_line_chart, mo, pl):
    df_diff = (
        df.with_columns(
            Passengers_diff = pl.col('Passengers').diff(diff_order.value)
        )
    )

    chart_diff = january_labeled_monthly_line_chart(
        df_diff,
        y="Passengers_diff",
        title=f"Passengers by Month after differencing of order {diff_order.value}",
    )

    mo.show_code()
    return chart_diff, df_diff


@app.cell
def _(mo):
    diff_order = mo.ui.slider(
        start=0,
        stop=5,
        step=1,
        value=1,
        show_value=True,
        label="Differencing order (k)",
    )

    diff_order
    return (diff_order,)


@app.cell
def _(chart_diff, mo):
    chart_diff_select = mo.ui.altair_chart(chart_diff)

    chart_diff_select
    return (chart_diff_select,)


@app.cell
def _(chart_diff_select, df_diff, mo, selection_summary_md):
    _df_diff_selected = chart_diff_select.apply_selection(df_diff)

    mo.md(
        selection_summary_md(
            _df_diff_selected,
            value_column="Passengers_diff",
            value_label="differenced passengers",
        )
    ).center()
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Log transforming to make the standard deviation constant

    The plot shows the standard deviation is still not constant through time. We can correct that by transforming the values using a log function.

    Because the log function behaves in a bad way when applied to values near zero, let's do the logging **before** the differencing.

    Select a time range in the graph below to see the statistics for it.
    """)
    return


@app.cell
def _(df, diff_order, january_labeled_monthly_line_chart, mo, np, pl):
    df_log_diff = (
        df.with_columns(
            Passengers_log_diff = np.log(pl.col("Passengers")).diff()
        )
    )

    chart_log_diff = january_labeled_monthly_line_chart(
        df_log_diff,
        y="Passengers_log_diff",
        title=f"Passengers after log transform and differencing "
            f"of order {diff_order.value}",
    )

    mo.show_code()
    return chart_log_diff, df_log_diff


@app.cell
def _(chart_log_diff, mo):
    chart_log_diff_select = mo.ui.altair_chart(chart_log_diff)

    chart_log_diff_select
    return (chart_log_diff_select,)


@app.cell
def _(chart_log_diff_select, df_log_diff, mo, selection_summary_md):
    _df_log_diff_selected = chart_log_diff_select.apply_selection(df_log_diff)

    mo.md(
        selection_summary_md(
            _df_log_diff_selected,
            value_column="Passengers_log_diff",
            value_label="differenced log passengers",
        )
    ).center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### The Box-Cox transform

    The log transform we used above is just a specific case of the more general **Box-Cox transform**:
    $$
    y_{i}^{(\lambda )}={\begin{cases}{\dfrac {y_{i}^{\lambda }-1}{\lambda }}&{\text{if }}\lambda \neq 0,\\\ln y_{i}&{\text{if }}\lambda =0,\end{cases}}
    $$

    The transformation depends on the value of a parameter $\lambda$. The scipy function that computes the Box-Cox transform finds the value of $\lambda$ that maximizes the log-likelihood function and returns it as the second output argument. [See how to use it in the documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html). If a simple log transform does not make the series stationary, try running this function.

    For our example, the log transform was enough. If we run the `boxcox` function, we find it picks a value of $\lambda$ that is close to zero:
    """)
    return


@app.cell
def _(boxcox, df, mo):
    _, lbd = boxcox(df['Passengers'])
    mo.show_code()
    return (lbd,)


@app.cell
def _(lbd, mo):
    mo.md(fr"""
    Value of $\lambda$ chosen by `boxcox` function: {lbd:.2f}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Decomposition

    We can try to decompose a time series into 3 components:

    - A **trend** component $T$,
    - A **seasonality** component $S$,
    - A **residual** component $R$.

    We can choose to combine these in an **additive model**
    $$
    y_t = T_t + S_t + R_t
    $$
    or in a **multiplicative model**
    $$
    y_t = T_t \cdot S_t \cdot R_t
    $$

    The choice of model depends on the characteristics of the realization of the time series we have. In our example, the fact that the standard deviation grows with time suggests the multiplicative model might be adequate:
    """)
    return


@app.cell
def _(df, mo, seasonal_decompose):
    decomposition_mult = seasonal_decompose(
        df['Passengers'], 
        model='multiplicative',
        period = 12
    )
    mo.show_code(decomposition_mult.plot())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Equivalently, we could have applied a log (or Box-Cox) transform to the `Passengers` column and then used an additive model. This is left as an exercise.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Autocorrelation

    The **correlation** between two vectors $X$ and $Y$ is a measure of how the values of $X$ change as the values of $Y$ change, as we traverse $X$ and $Y$ in order.

    If the values of $Y$ are approximately proportional to the values of $X$, then the correlation is close to $1$.

    If the values of $Y$ are approximately inversely proportional to the values of $X$, then the correlation is close to $-1$.

    Intermediate situations produce values between $-1$ and $1$ for the correlation. A value of zero means there is no **linear** correlation between $X$ and $Y$.

    In the analysis of time series, it is useful to measure the **autocorrelation** between the series and **lagged versions** of the series. The result for our example can be visualized as the plot below. The dot corresponding to position $k$ in the $x$ axis is the value of the correlation between the series $Y_t$ and the lagged series $Y_{t - k}$.

    The blue region indicates a $95\%$ confidence interval for the null correlation; i.e., only dots outside it represent significant correlations.
    """)
    return


@app.cell
def _(df, mo, plot_acf):
    mo.show_code(plot_acf(df['Passengers'], lags=48))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As expected, a cycle of length $12$ can be observed, corresponding to yearly patterns.

    We can also see that lags up to $k = 14$ are significantly correlated to the series. This means that, in an autoregressive model (to be discussed below), we should use an equation of the form
    $$
    y_t = \varphi_1 y_{t-1} + \varphi_2 y_{t-2} + \cdots + \varphi_{14} y_{t-14} + \varepsilon_t
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Modeling and Forecasting

    All we have so far are the **data** for the numbers of passengers. These data can be considered to be a **realization** of a time series. Using this terminology, the time series itself is the **process** that produces the numbers as time goes by, according to certain rules that we do not know but that **we want to model**.

    In other words, we want to find a **model for the time series**: a mathematical expression (i.e. an equation of the form $y_t = \cdots$) that captures the effects of the phenomena that produce these numbers.

    These models can be of different forms.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Autoregressive models

    This kind of model tries to see the current value $y_t$ as a function of the previous values $y_{t-1}, y_{t-2}, \ldots, y_{t-p}$. The number $p$ of previous values is the **order** of the autoregression.

    The equation is
    $$
    y_t = \varphi_1y_{t-1} + \varphi_2y_{t-2} + \cdots + \varphi_py_{t-p} + \varepsilon_t
    $$

    Note that

    - If all coefficients are zero, this is **white noise**.
    - If $p = 1$ and $\varphi_1 = 1$, this is a **random walk**.

    As with any statistical model, the coefficients $\varphi_i$ must be **estimated** from the **training data** we have. Once we have values for the coefficients, we **validate** the model against the **test data** we set aside for that purpose.

    In order to build an autoregressive model, the data we have must be **stationary**. If necessary, we can apply the transformations presented at the beginning of this document.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Moving-average models

    This kind of model tries to express $y_t$ as a function of previous **errors** (instead of previous **values**). More precisely, the equation is
    $$
    y_{t} = \mu +\varepsilon_{t} + \theta_{1}\varepsilon_{t-1} + \cdots + \theta_{q}\varepsilon _{t-q}
    $$
    where $\mu$ is the mean of the series.

    The number $q$ of lagged errors used is called the **order** of the model.

    This model is more complicated to build, because the lagged error terms are not observable and must be estimated too, along with the coefficients $\theta_i$.

    Again, the data must be stationary for this model to be used.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ARIMA models

    ARIMA stands for **autoregressive integrated moving average**. This kind of model uses a combination of three of the techniques we have already covered here:

    1. Auto regression (AR),
    2. Differencing, also called integration (I),
    3. Moving average (MA).

    In order to build an ARIMA model, we must choose the orders for the three methods:

    - $p$: the order for the autoregressive model,
    - $d$: the order for the differencing,
    - $q$: the order for the moving-average model.

    After these choices are made, the model is referred to as ARIMA($p$, $d$, $q$).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fitting a model to the passenger data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Make the standard deviation constant

    ARIMA takes care of the differencing, but we must apply a transformation to make the standard deviation constant. Let's use the Box-Cox transform:
    """)
    return


@app.cell
def _(boxcox, df, mo):
    passengers, lamb = boxcox(df['Passengers'])
    mo.show_code()
    return lamb, passengers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Split the data

    Let's train the model on the first 80% of the data and test it on the remaining 20%:
    """)
    return


@app.cell
def _(mo, passengers):
    npoints = len(passengers)
    train = passengers[:int(.8 * npoints)]
    test = passengers[int(.8 * npoints):]
    mo.show_code()
    return test, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Choose the orders

    Based on the autocorrelation of the data, $p = 15$ is a good choice for the order of the autoregression.

    We will choose $d = 1$ for the differencing.

    We will choose $q = 15$ for the moving-average model.
    """)
    return


@app.cell
def _(mo):
    p = 15
    d = 1
    q = 15
    mo.show_code()
    return d, p, q


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Fit the model on the training data
    """)
    return


@app.cell
def _(ARIMA, d, mo, p, q, train):
    model = ARIMA(train, order=(p, d, q)).fit()
    mo.show_code()
    return (model,)


@app.cell
def _(mo, model):
    mo.show_code(model.summary())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Test the model on the test data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We ask the model to predict the next `len(test)` values of the series. The returned values are based on the Box-Cox-transformed data, so we must undo the transform.
    """)
    return


@app.cell
def _(inv_boxcox, lamb, mo, model, pl, test):
    boxcox_forecasts = model.forecast(len(test))
    forecasts = pl.Series(inv_boxcox(boxcox_forecasts, lamb))
    mo.show_code()
    return (forecasts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now build a plot to compare the forecasts to the real values in the test data. We show the known values in blue (starting a year before the beginning of the test data) and the predicted values for the test data (in red):
    """)
    return


@app.cell
def _(df, forecasts, january_labeled_monthly_line_chart, mo, test, train):
    # Build a df only for the forecast months
    df_forecasts = df.clone().slice(len(train)).with_columns(
        Passengers = forecasts
    )

    df_known = df.slice(-len(test) - 12)

    chart_known = january_labeled_monthly_line_chart(
        df_known,
        y="Passengers",
        title="Passengers by Month",
        line_color='blue',
        show_points = False
    )

    chart_forecast = january_labeled_monthly_line_chart(
        df_forecasts,
        y="Passengers",
        title="Passengers by Month",
        line_color='red',
        show_points = False
    )

    mo.show_code(chart_known + chart_forecast)
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Model evaluation

    We can choose the best value for the orders $p$, $d$ and $q$ by examining the autocorrelation in the data, as shown in a previous section, or by trying several values and evaluating the results.

    In a future document, we will see how we can evaluate the quality of the models we have built.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## References

    ### Marimo

    - https://docs.marimo.io/

    ### Polars

    - Book: https://scholarfriends.com/singlePaper/623454/ebook-pdf-python-polars-the-definitive-guide-by-jeroen-janssens-thijs-nieuwdorp
    - Site: https://docs.pola.rs/api/python/stable/reference/index.html

    ### Altair

    - https://altair-viz.github.io/index.html

    ### Time series

    - Book: Montgomery, D.C. and Jennings, C.L. and Kulahci, M. (2024). *Introduction to Time Series Analysis and Forecasting*. Wiley.
    - Video playlist: https://youtube.com/playlist?list=PLvcbYUQ5t0UHOLnBzl46_Q6QKtFgfMGc3
    - Video playlist: https://www.youtube.com/playlist?list=PLKmQjl_R9bYd32uHImJxQSFZU5LPuXfQe
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Helper functions and misc code
    """)
    return


@app.cell
def _(alt, mo, pl):
    def january_labeled_monthly_line_chart(
        df: pl.DataFrame,
        x: str = "Month",
        y: str = "Passengers",
        *,
        january_label_format: str = "%Y-%m",
        label_angle: int = 45,
        line_color: str = 'blue',
        show_points: bool = True,
        grid_color: str = "#aaaaaa",
        grid_opacity: float = .8,
        grid_width: float = .8,
        width: int | str | None = 'container',
        height: int | str | None = None,
        title: str | None = None,
    ) -> alt.Chart:
        jan = df.filter(pl.col(x).dt.month() == 1)

        line = (
            alt.Chart(df, title=title)
            .mark_line(point=show_points, color=line_color)
            .encode(
                x=alt.X(
                    f"{x}:T",
                    axis=alt.Axis(
                        grid=False,
                        tickCount={"interval": "month", "step": 1},
                        labelExpr=(
                            f"month(datum.value) == 0 "
                            f"? timeFormat(datum.value, '{january_label_format}') "
                            f": ''"
                        ),
                        labelAngle=label_angle,
                    ),
                ),
                y=alt.Y(
                    f"{y}:Q",
                    axis=alt.Axis(
                        grid=True,
                        gridColor=grid_color,
                        gridOpacity=grid_opacity,
                        gridWidth=grid_width,
                    ),
                ),
            )
        )

        jan_rules = (
            alt.Chart(jan)
            .mark_rule(
                color=grid_color,
                opacity=grid_opacity,
                strokeWidth=grid_width,
            )
            .encode(
                x=alt.X(f"{x}:T")
            )
        )

        chart = jan_rules + line

        if width is not None:
            chart = chart.properties(width=width)
        if height is not None:
            chart = chart.properties(height=height)

        return chart

    mo.show_code()
    return (january_labeled_monthly_line_chart,)


@app.cell
def _(mo, pl):
    def selection_summary_md(
        df: pl.DataFrame,
        *,
        value_column: str,
        value_label: str,
    ) -> str:
        _summary = df.select(
            pl.col("Month").min().alias("month_min"),
            pl.col("Month").max().alias("month_max"),
            pl.len().alias("n_selected_points"),
            pl.col(value_column).mean().alias("mean_value"),
            pl.col(value_column).std().alias("std_value"),
        ).row(0, named=True)

        _month_min = _summary["month_min"]
        _month_max = _summary["month_max"]
        _n_selected_points = _summary["n_selected_points"]
        _mean_value = _summary["mean_value"]
        _std_value = _summary["std_value"]

        _mean_display = "N/A" if _mean_value is None else f"{_mean_value:.2f}"
        _std_display = "N/A" if _std_value is None else f"{_std_value:.2f}"

        return f"""
        Minimum month: `{_month_min}`  
        Maximum month: `{_month_max}`  
        Selected points: `{_n_selected_points}`  
        Mean {value_label}: `{_mean_display}`  
        Std. dev. {value_label}: `{_std_display}`
        """

    mo.show_code()
    return (selection_summary_md,)


if __name__ == "__main__":
    app.run()
