"""Monte Carlo simulation of the Dunning-Kruger experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import altair as alt
import numpy as np
import polars as pl
import streamlit as st


if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt


QUARTILES = ("Bottom", "2nd", "3rd", "Top")


def percentile(x: npt.NDArray[np.float_]) -> npt.NDArray[np.int_]:
    return x.argsort().argsort() * 100 // len(x)

def generate_data(
    n_participants: int,
    corr_coef: float,
    random_seed: int,
) -> pl.DataFrame:
    rng = np.random.default_rng(random_seed)
    test_score = rng.normal(size=n_participants)
    perceived_ability = (
        corr_coef * test_score +
        np.sqrt(1 - corr_coef * corr_coef) * rng.normal(size=n_participants)
    )

    return (
        pl.DataFrame({
            "test_score_percentile": percentile(test_score),
            "perceived_ability_percentile": percentile(perceived_ability),
        })
        .with_columns(
            pl.col("test_score_percentile")
                .qcut(4, labels=QUARTILES)
                .alias("test_score_quartile"),
            pl.col("perceived_ability_percentile")
                .qcut(4, labels=QUARTILES)
                .alias("perceived_ability_quartile"),
        )
    )


def create_percentile_chart(data: pl.DataFrame) -> alt.Chart:
    return alt.Chart(data).mark_point().encode(
        alt.X("test_score_percentile:Q").title("test score percentile"),
        alt.Y("perceived_ability_percentile:Q").title("perceived ability percentile"),
    )


def create_quartile_chart(data: pl.DataFrame, quartile_col: str) -> alt.Chart:
    return (
        data.select(
            quartile_col,
            pl.col("test_score_percentile").alias("test score"),
            pl.col("perceived_ability_percentile").alias("perceived ability"),
        )
        .group_by(quartile_col)
        .mean()
        .melt(id_vars=quartile_col, value_name="average")
        .pipe(alt.Chart)
        .mark_line(point=True)
        .encode(
            alt.Color("variable:N")
                .sort(("test score", "perceived ability"))
                .legend(orient="bottom-right")
                .title(None),
            alt.X(f"{quartile_col}:N")
                .sort(QUARTILES)
                .axis(labelAngle=0)
                .title(quartile_col.replace("_", " ")),
            alt.Y("average:Q")
                .scale(domain=(0, 100))
                .title("average percentile"),
        )
    )


def custom_theme() -> dict[str, Any]:
    return {
        "config": {
            "axis": {
                "grid": False,
                "labelColor": "#7F7F7F",
                "labelFontSize": 14,
                "tickColor": "#7F7F7F",
                "titleColor": "#7F7F7F",
                "titleFontSize": 16,
                "titleFontWeight": "normal",
            },
            "legend": {
                "labelColor": "#7F7F7F",
                "labelFontSize": 14,
            },
            "view": {
                "height": 320,
                "width": 480,
                "stroke": False,
            },
        },
    }


if __name__ == "__main__":
    alt.themes.register("custom_theme", custom_theme)
    alt.themes.enable("custom_theme")

    with st.sidebar:
        st.title("Monte Carlo simulation of the Dunning-Kruger experiment")

        n_participants = st.slider(
            label="Number of participants",
            min_value=50,
            max_value=150,
            value=100,
            step=10,
        )

        corr_coef = st.slider(
            label="Correlation",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
        )

        random_seed = st.number_input(label="Random seed", value=42)

    data = generate_data(
        n_participants=n_participants,
        corr_coef=corr_coef,
        random_seed=random_seed,  # type: ignore
    )

    st.header("Test score vs. perceived ability")
    st.altair_chart(
        create_percentile_chart(data),
        theme=None,
    )

    st.header("Average percentiles by test score quartiles")
    st.altair_chart(
        create_quartile_chart(data, "test_score_quartile"),
        theme=None,
    )

    st.header("Average percentiles by perceived ability quartiles")
    st.altair_chart(
        create_quartile_chart(data, "perceived_ability_quartile"),
        theme=None,
    )
