import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Time series
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


@app.cell
def _(mo):
    mo.md(r"""
    ### Example: airline passenger data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Imports:
    """)
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    alt.data_transformers.enable('marimo_csv')
    from scipy.stats import boxcox
    import numpy as np
    mo.show_code()
    return alt, mo, np, pl


@app.cell
def _(mo):
    mo.md(r"""
    As our main example, we will use data about the number of passengers in flights from 1949 to 1960. The file is available online [at the Kaggle site](https://www.kaggle.com/datasets/ashfakyeafi/air-passenger-data-for-time-series-analysis).
    """)
    return


@app.cell
def _(mo, pl):
    df = pl.read_csv('AirPassengers.csv').with_columns(
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
    #### Differencing for the mean

    Let's write $y(t)$ for the number of passengers at time $t$. Instead of working with the values
    $$
    y(t)
    $$
    we work with the **discrete differences**
    $$
    y(t) - y(t-k)
    $$
    for some natural $k > 0$. The value of $k$ is the **order** of the differencing.

    Usually, $k = 1$ is enough to make the mean constant through time.

    The `diff()` method (in Pandas or in Polars) computes the discrete difference of a series (which is also a series).

    Choose the value of $k$ in the slider below to see how the graph below changes, then select a time range in the graph to see the statistics for it.
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
    #### Log transforming for the standard deviation

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

    For our example, the log transform was enough.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Decomposition

    We can try to decompose a time series into 3 components:

    - Trend,
    - Seasonality,
    - Residuals.

    Going back to the original time series (with no transformations), we get the following components:
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Autocorrelation
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Forecasting
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Model selection
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## References
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
        grid_color: str = "#aaaaaa",
        grid_opacity: float = .8,
        grid_width: float = .8,
        width: int | None = None,
        height: int | None = None,
        title: str | None = None,
    ) -> alt.Chart:
        jan = df.filter(pl.col(x).dt.month() == 1)

        line = (
            alt.Chart(df, title=title)
            .mark_line(point=True)
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
