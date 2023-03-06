import dash
from dash import html
import dash_bootstrap_components as dbc
from dash import dcc, dash_table, Dash, ctx
from dash.dependencies import Output, Input, State, MATCH, ALL, ALLSMALLER
import pyarrow, os
import pandas as pd

# import base64
# import io
from app import app
from DataLoader import dataloader_layout, load_data_for_other_tabs
from Tab1 import tab1_func_to_get_the_basetabdata
from Tab2 import tab2_func_to_get_the_basetabdata


# app = dash.Dash(
#     __name__,
#     external_stylesheets=[dbc.themes.MORPH],
#     suppress_callback_exceptions=True,
# )



app_tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(
                    label="Data Loader",
                    tab_id="data-loader",
                    labelClassName="text-success font-weight-bold",
                    activeLabelClassName="text-danger",
                ),
                dbc.Tab(
                    label="Tab1",
                    tab_id="tab1",
                    labelClassName="text-success font-weight-bold",
                    activeLabelClassName="text-danger",
                ),
                dbc.Tab(
                    label="Tab2",
                    tab_id="tab2",
                    labelClassName="text-success font-weight-bold",
                    activeLabelClassName="text-danger",
                ),
            ],
            id="tabs",
            active_tab="data-loader",
        ),
    ],
    className="mt-3",
)

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("Anomaly Detection Dashboard", style={"textAlign": "center"}),
                width=12,
            )
        ),
        html.Hr(),
        dbc.Row(dbc.Col(app_tabs, width=12), className="mb-3"),
        html.Div(id="content", children=[]),
    ],
    style={"background-color": "#DBF9FC"},
)


@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def switch_tab(
    tab_chosen,
):
    if tab_chosen == "data-loader":
        return dataloader_layout
    elif tab_chosen == "tab1":
        return tab1_func_to_get_the_basetabdata(load_data_for_other_tabs())
        # load_data_for_other_tabs()
        # return tab1_layout
    elif tab_chosen == "tab2":
        return tab2_func_to_get_the_basetabdata(load_data_for_other_tabs())
    return html.P("This shouldn't be displayed for now...")


if __name__ == "__main__":
    app.run_server(debug=False)
