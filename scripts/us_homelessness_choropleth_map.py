#!/usr/bin/env python3

import argparse
import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create a choropleth map of homelessness rates by state"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="complete_dataset/AHAR_2024_homelessness_table.csv",
        help="Path to the AHAR state table CSV",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        required=True,
        help="Output path stem (e.g., output/charts/ahar/us_homelessness_rate_map)",
    )
    parser.add_argument(
        "--show_county_names",
        action="store_true",
        help="Show county names as text labels next to markers",
    )
    parser.add_argument(
        "--show_numbers_with_names",
        action="store_true",
        help="(Deprecated; now default) Show 1-5/A-E inside dots and draw a keyed list on the left",
    )
    parser.add_argument(
        "--no_key",
        action="store_true",
        help="Disable the left-side key and in-marker labels (default is to show them)",
    )
    return parser.parse_args()


def create_map(data, save_name, show_county_names=False, show_numbers_with_names=False):
    state_data = data.copy()
    state_data["text"] = state_data["homeless_rate"].round(1).astype(str)

    save_stem = Path(save_name)
    save_stem.parent.mkdir(parents=True, exist_ok=True)
    state_data.to_csv(f"{save_stem}.csv", index=False)

    title_x = 0.5

    fig = go.Figure(
        data=go.Choropleth(
            locations=state_data["state_abbr"],
            z=state_data["homeless_rate"],
            locationmode="USA-states",
            colorscale="Reds",
            coloraxis="coloraxis",
            text=state_data["text"],
            hovertemplate="<b>%{location}</b><br>Rate: %{z:.1f}<extra></extra>",
        )
    )

    locations_sf = [
        dict(
            lat=37.7749,
            lon=-122.4194,
            name="San Francisco County, CA",
            number="1",
            offset_lat=0.0,
            offset_lon=0.0,
        ),
        dict(
            lat=45.5146,
            lon=-122.6587,
            name="Multnomah County, OR",
            number="2",
            offset_lat=0.0,
            offset_lon=0.0,
        ),
        dict(
            lat=42.6795,
            lon=-78.7561,
            name="Erie County, NY",
            number="3",
            offset_lat=0.0,
            offset_lon=0.0,
        ),
        dict(
            lat=39.4648,
            lon=-76.7337,
            name="Baltimore County, MD",
            number="4",
            offset_lat=0.0,
            offset_lon=0.0,
        ),
        dict(
            lat=31.8040,
            lon=-106.2051,
            name="El Paso County, TX",
            number="5",
            offset_lat=0.0,
            offset_lon=0.0,
        ),
    ]

    locations_b = [
        dict(
            lat=41.6381,
            lon=-86.2595,
            name="St. Joseph County, IN",
            letter="A",
            offset_lat=-0.5,
            offset_lon=-0.2,
        ),
        dict(
            lat=42.3121,
            lon=-89.1606,
            name="Winnebago County, IL",
            letter="B",
            offset_lat=0.0,
            offset_lon=0.0,
        ),
        dict(
            lat=42.2917,
            lon=-85.5908,
            name="Kalamazoo County, MI",
            letter="C",
            offset_lat=0.5,
            offset_lon=0.2,
        ),
        dict(
            lat=41.4421,
            lon=-75.5742,
            name="Lackawanna County, PA",
            letter="D",
            offset_lat=0.0,
            offset_lon=0.0,
        ),
        dict(
            lat=35.9308,
            lon=-94.1514,
            name="Washington County, AR",
            letter="E",
            offset_lat=0.0,
            offset_lon=0.0,
        ),
    ]

    def add_markers(locations, color, symbol, label_key="number"):
        for loc in locations:
            marker_lon = loc["lon"] + loc.get("offset_lon", 0.0)
            marker_lat = loc["lat"] + loc.get("offset_lat", 0.0)

            if show_numbers_with_names:
                label_text = loc.get(label_key, "")
                marker_size = 30
                fig.add_trace(
                    go.Scattergeo(
                        lon=[marker_lon],
                        lat=[marker_lat],
                        text=[label_text],
                        mode="markers+text",
                        marker=dict(
                            size=marker_size,
                            color=color,
                            line=dict(width=1.5, color="black"),
                            symbol=symbol,
                        ),
                        textposition="middle center",
                        textfont=dict(size=24, color="black"),
                        showlegend=False,
                    )
                )
            elif show_county_names:
                text_lon = marker_lon + loc.get("offset_lon", 0.0) * 2
                text_lat = marker_lat + loc.get("offset_lat", 0.0) * 2
                fig.add_trace(
                    go.Scattergeo(
                        lon=[text_lon, marker_lon],
                        lat=[text_lat, marker_lat],
                        mode="lines",
                        line=dict(width=2, color="black"),
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scattergeo(
                        lon=[text_lon],
                        lat=[text_lat],
                        text=[loc["name"]],
                        mode="text",
                        textfont=dict(size=14, color="black"),
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scattergeo(
                        lon=[marker_lon],
                        lat=[marker_lat],
                        mode="markers",
                        marker=dict(
                            size=30,
                            color=color,
                            line=dict(width=1.5, color="black"),
                            symbol=symbol,
                        ),
                        showlegend=False,
                    )
                )
            else:
                fig.add_trace(
                    go.Scattergeo(
                        lon=[marker_lon],
                        lat=[marker_lat],
                        mode="markers",
                        marker=dict(
                            size=30,
                            color=color,
                            line=dict(width=1.5, color="black"),
                            symbol=symbol,
                        ),
                        showlegend=False,
                    )
                )

    add_markers(locations_sf, "#8ff0a4", "circle", "number")
    add_markers(locations_b, "#f9f06b", "square", "letter")

    if show_numbers_with_names:
        key_text = "<b>San Francisco Cluster:</b><br>"
        for loc in locations_sf:
            key_text += f"{loc['number']}. {loc['name']}<br>"
        key_text += "<br><b>St. Joseph Cluster:</b><br>"
        for loc in locations_b:
            key_text += f"{loc['letter']}. {loc['name']}<br>"

        fig.add_annotation(
            text=key_text,
            xref="paper",
            yref="paper",
            x=-0.12,
            y=0.4,
            xanchor="left",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=10,
        )

    fig.add_trace(
        go.Scattergeo(
            lon=[None],
            lat=[None],
            mode="markers",
            marker=dict(size=30, color="#8ff0a4", line=dict(width=1.5, color="black")),
            name="San Francisco Cluster",
        )
    )

    fig.add_trace(
        go.Scattergeo(
            lon=[None],
            lat=[None],
            mode="markers",
            marker=dict(
                size=30,
                color="#f9f06b",
                line=dict(width=1.5, color="black"),
                symbol="square",
            ),
            name="St. Joseph Cluster",
        )
    )

    fig.update_layout(
        title=dict(
            text="Homelessness Rates Across the U.S. (per 10,000 People)",
            font=dict(size=40, color="black"),
            y=0.95,
            x=title_x,
            xanchor="center",
            yanchor="top",
        ),
        geo=dict(
            scope="usa",
            projection=go.layout.geo.Projection(type="albers usa"),
            showlakes=True,
            lakecolor="white",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            showcoastlines=True,
            coastlinecolor="gray",
            showframe=False,
        ),
        legend=dict(
            yanchor="bottom",
            y=-0.12 if show_numbers_with_names else -0.05,
            xanchor="left",
            x=0.7 if show_numbers_with_names else 0.99,
            font=dict(size=30, color="black"),
        ),
        margin=dict(
            l=280 if show_numbers_with_names else 20,
            r=20,
            t=100,
            b=60 if show_numbers_with_names else 40,
        ),
        coloraxis=dict(
            colorscale="Reds",
            colorbar=dict(
                tickfont=dict(size=27, color="black"),
                tickvals=[0, 20, 40, 60, 80],
                title=None,
                thickness=20,
                len=0.5,
                orientation="h",
                x=title_x,
                y=0.92,
                xanchor="center",
                yanchor="bottom",
            ),
        ),
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=True,
        hovermode="closest",
        modebar=dict(
            remove=[
                "zoom",
                "pan",
                "select",
                "lasso",
                "zoomIn",
                "zoomOut",
                "autoScale",
                "resetScale",
            ]
        ),
        font=dict(family="Arial, sans-serif"),
    )

    print("Saving outputs...")
    img_width = 1480 if show_numbers_with_names else 1200
    img_height = 830

    fig.write_html(f"{save_stem}.html", config={"displayModeBar": False})
    fig.write_image(f"{save_stem}.png", width=img_width, height=img_height, scale=2)

    img_bytes = fig.to_image(
        format="png", width=img_width, height=img_height, scale=10
    )
    img = Image.open(io.BytesIO(img_bytes))
    plt.figure(figsize=(img_width / 100, img_height / 100))
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(f"{save_stem}.pdf", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    fig.write_image(f"{save_stem}.svg", width=img_width, height=img_height, scale=2)
    print("Done!")


def main():
    args = parse_arguments()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {data_path}\n"
            "Provide --data_path pointing to your AHAR state table CSV."
        )
    data = pd.read_csv(data_path)

    state_to_abbreviation = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "District of Columbia": "DC",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY",
    }

    data["state_abbr"] = data["State"].map(state_to_abbreviation)
    data.rename(
        columns={
            "Number of People in State Experiencing Homelessness per 10,000 People": "homeless_rate"
        },
        inplace=True,
    )

    # Default behavior: show the key + labels unless explicitly disabled.
    show_numbers_with_names = (not args.no_key) or args.show_numbers_with_names

    create_map(
        data,
        args.save_name,
        show_county_names=args.show_county_names,
        show_numbers_with_names=show_numbers_with_names,
    )


if __name__ == "__main__":
    main()

