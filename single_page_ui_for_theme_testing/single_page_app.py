import dash
from dash import html
import dash_bootstrap_components as dbc
from dash import dcc, dash_table, Dash, ctx
from dash.dependencies import Output, Input, State, MATCH, ALL, ALLSMALLER
import pyarrow, os
import pandas as pd

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    # meta_tags=[
    #     {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    # ],
)

button_group_data_load_options = html.Div(
    [
        dbc.RadioItems(
            id="radios",
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "data/raw", "value": "data/raw"},
                {"label": "Local", "value": "Local"},
                {"label": "Cloud", "value": "Cloud"},
            ],
            value="data/raw",
            # style={"padding-left": "0"},
            style={"paddingLeft": "0"},
        ),
        html.Div(id="output"),
    ],
    className="radio-group",
    style={"formCheck ": "padding-left: 0"},
)


modal = html.Div(
    [
        dbc.Button("Open modal", id="open", n_clicks=0),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Header")),
                dbc.ModalBody("This is the content of the modal"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close", className="ms-auto", n_clicks=0)
                ),
            ],
            id="modal",
            is_open=False,
        ),
    ]
)


@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


right_side_layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Dataloader options"),
                        button_group_data_load_options,
                        html.Hr(),
                        dcc.Checklist(
                            ["New York City", "Montréal", "San Francisco"],
                            ["New York City", "Montréal"],
                            inline=True,
                        ),
                        html.Hr(),
                        html.Label("Open a Modal by clicking the button"),
                        modal,
                        html.Hr(),
                        html.Button("Download Text", id="btn-download-txt"),
                        dcc.Download(id="download-text"),
                        html.Hr(),
                        html.H6("Input Box - Number"),
                        dcc.Input(
                            id="input_",
                            type="number",
                            placeholder="input type",
                        ),
                        html.Br(),
                        html.H6("Input Box - Text"),
                        dcc.Input(
                            id="input_1",
                            type="text",
                            placeholder="input type",
                        ),
                        html.Hr(),
                        dcc.RangeSlider(0, 20, 1, value=[5, 15], id="my-range-slider"),
                    ]
                )
            ]
        )
    ]
)

left_card_main = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H6("Selection Panel", className="card-title"),
                html.P(
                    "Select Dimensions to Filter",
                    className="card-text",
                ),
                dcc.Dropdown(
                    id="filter-dropdwn",
                    options=["option1", "option2", "option3"],
                    value="",
                    clearable=True,
                    multi=True,
                    style={"color": "#000000"},
                ),
                html.Br(),
                html.Span(
                    [
                        html.Button("Add Filter", id="add-filter-val", n_clicks=0),
                        html.Spacer(),
                        html.Button("Reset Filter", id="reset-filter-val", n_clicks=0),
                    ]
                ),
                html.Br(),
                html.Div(
                    id="Filter-subsection-placeholder",
                    children=[],
                ),
                html.Br(),
                # dcc.Dropdown(
                #     id="user_choice2",
                #     options=["Input 1", "Input 2", "Input 3"],
                #     value="Input 1",
                #     clearable=False,
                #     style={"color": "#000000"},
                # ),
                html.Button("Update Data", id="submit-val", n_clicks=0),
            ]
        ),
    ],
    color="dark",  # https://bootswatch.com/default/ for more card colors
    inverse=True,  # change color of text (black or white)
    outline=False,  # True = remove the block colors from the background and header
)

tab1_content = dbc.Container(
    [
        dbc.Row(html.H3("Tab1")),
        dbc.Row([dbc.Col(left_card_main, width=3), dbc.Col(right_side_layout)]),
    ]
)

app_tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(
                    label="Data Loader",
                    tab_id="data-loader",
                    labelClassName="text-success font-weight-bold",
                    activeLabelClassName="text-danger",
                    children=[tab1_content],
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
    # style={"background-color": "#DBF9FC"},
)

if __name__ == "__main__":
    app.run_server(debug=False)
