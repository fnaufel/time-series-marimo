import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


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
    mo.md("""
    <div style="text-align: right;">
    <em>It is difficult to make predictions, especially about the future.</em>
    <br />
    NIELS BOHR, Danish physicist
    </div>
    """)
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
    mo.show_code()
    return alt, mo, pl


@app.cell
def _(mo):
    mo.md(r"""
    As our main example, we will use data about the number of passengers in flights from 1949 to 1960. The file is available online [at the Kaggle site](https://www.kaggle.com/datasets/ashfakyeafi/air-passenger-data-for-time-series-analysis).
    """)
    return


@app.cell
def _(mo, pl):
    df = pl.read_csv('AirPassengers.csv').with_columns(
        (pl.col("Month") + "-01").str.to_date("%Y-%m-%d").alias("Month")
    )

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
    chart = january_labeled_monthly_line_chart(
        df,
        y="#Passengers",
        title="Passengers by Month",
    )

    mo.show_code(chart)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Stationarity
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Seasonality
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Decomposition
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Switch to editor mode** to see the code below.
    """)
    return


@app.cell
def _(alt, pl):
    def january_labeled_monthly_line_chart(
        df: pl.DataFrame,
        x: str = "Month",
        y: str = "#Passengers",
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

    return (january_labeled_monthly_line_chart,)


if __name__ == "__main__":
    app.run()
