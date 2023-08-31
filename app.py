"""Monte Carlo simulation of the Dunning-Kruger experiment."""

import altair as alt
import numpy as np
import polars as pl
import streamlit as st


QUARTILES = ("Bottom", "2nd", "3rd", "Top")


def generate_data(
    corr_coef: float,
    n_participants: int,
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
            "test_score": test_score,
            "perceived_ability": perceived_ability,
        })
        .with_columns(
            pl.col("test_score").rank()
                .mul(100).floordiv(n_participants)
                .alias("test_score_percentile"),
            pl.col("perceived_ability").rank()
                .mul(100).floordiv(n_participants)
                .alias("perceived_ability_percentile"),
            pl.col("test_score").qcut(4, labels=QUARTILES)
                .alias("test_score_quartile"),
            pl.col("perceived_ability").qcut(4, labels=QUARTILES)
                .alias("perceived_ability_quartile"),
        )
    )


def create_percentile_chart(data: pl.DataFrame) -> alt.Chart:
    return alt.Chart(data).mark_point().encode(
        alt.X("test_score_percentile:Q"),
        alt.Y("perceived_ability_percentile:Q"),
    )


def create_quartile_chart(data: pl.DataFrame, quartile_col: str) -> alt.Chart:
    return (
        data.select(
            quartile_col,
            "test_score_percentile",
            "perceived_ability_percentile",
        )
        .group_by(quartile_col)
        .mean()
        .melt(id_vars=quartile_col, value_name="average")
        .pipe(alt.Chart)
        .mark_line(point=True)
        .encode(
            alt.Color("variable:N").title(None)
                .sort(("test_score_percentile", "perceived_ability_percentile")),
            alt.X(f"{quartile_col}:O").sort(QUARTILES),
            alt.Y("average:Q").title(None),
        )
    )


if __name__ == "__main__":
    st.title("Monte Carlo simulation of the Dunning-Kruger experiment")
    st.header("Parameters")

    corr_coef = st.slider(
        label="Correlation coefficient",
        min_value=0.05,
        max_value=0.95,
        value=0.5,
        step=0.05,
    )

    n_participants = st.slider(
        label="Number of participants",
        min_value=100,
        max_value=500,
        value=300,
        step=20,
    )

    random_seed = st.number_input(label="Random seed", value=42)

    data = generate_data(
        corr_coef=corr_coef,
        n_participants=n_participants,
        random_seed=random_seed,  # type: ignore
    )

    st.header("Test score vs. perceived ability")
    st.altair_chart(
        create_percentile_chart(data),
        use_container_width=True,
        theme=None,
    )

    st.header("Average percentiles by test score quartiles")
    st.altair_chart(
        create_quartile_chart(data, "test_score_quartile"),
        use_container_width=True,
        theme=None,
    )

    st.header("Average percentiles by perceived ability quartiles")
    st.altair_chart(
        create_quartile_chart(data, "perceived_ability_quartile"),
        use_container_width=True,
        theme=None,
    )
