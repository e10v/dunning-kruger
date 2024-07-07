# ruff: noqa: RUF001
"""Random number simulation of the Dunning and Kruger experiments."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import altair as alt
import numpy as np
import polars as pl
import streamlit as st


if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt


QUARTILES = ("bottom", "2nd", "3rd", "top")


def percentile(x: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
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
            "test_score": test_score,
            "perceived_ability": perceived_ability,
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


def create_point_chart(data: pl.DataFrame, x: str, y: str) -> alt.Chart:
    return alt.Chart(data).mark_point().encode(
        alt.X(f"{x}:Q").title(x.replace("_", " ")),  # type: ignore
        alt.Y(f"{y}:Q").title(y.replace("_", " ")),  # type: ignore
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
        .unpivot(index=quartile_col, value_name="average")
        .pipe(alt.Chart)
        .mark_line(point=True)
        .encode(
            alt.Color("variable:N")
                .sort(("test score", "perceived ability"))  # type: ignore
                .legend(orient="bottom-right")
                .title(None),
            alt.X(f"{quartile_col}:N")
                .sort(QUARTILES)  # type: ignore
                .axis(labelAngle=0)
                .title(quartile_col.replace("_", " ")),
            alt.Y("average:Q")
                .scale(domain=(0, 100))
                .title("average percentile"),  # type: ignore
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
        st.header("Parameters")

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

    st.title("Random number simulation of the Dunning and Kruger experiments")
    st.markdown(textwrap.dedent("""\
        [![Source Code](https://img.shields.io/badge/source_code-green?logo=github&labelColor=gray)](https://github.com/e10v/dunning-kruger)
        [![Blog Post](https://img.shields.io/badge/blog_post-blue?label=e10v&labelColor=gray)](https://e10v.me/debunking-dunning-kruger-effect/)
    """))

    st.markdown(textwrap.dedent("""\
        The Dunning–Kruger effect is a cognitive bias wherein people with limited\
        competence in a particular domain overestimate their abilities. It turns out\
        that the Dunning and Kruger experiments do not prove that the effect is real.\
        This app illustrates this using a random number simulation. The data generation\
        process doesn't imply that "incompetent" participants overestimate their\
        abilities. See more details in my blog post [Debunking the Dunning–Kruger\
        effect with random number simulation](https://e10v.me/debunking-dunning-kruger-effect/).
    """))

    st.header("Test score vs. perceived ability")
    st.altair_chart(
        create_point_chart(data, x="test_score", y="perceived_ability"),
        theme=None,
    )

    st.header("Test score percentile vs. perceived ability percentile")
    st.altair_chart(
        create_point_chart(
            data,
            x="test_score_percentile",
            y="perceived_ability_percentile",
        ),
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
