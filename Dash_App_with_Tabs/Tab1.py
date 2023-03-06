import dash
from dash import html
import dash_bootstrap_components as dbc
from dash import dcc, dash_table, Dash, ctx
from dash.dependencies import Output, Input, State, MATCH, ALL, ALLSMALLER
import pyarrow, os
import pandas as pd
import base64
import io
from DataLoader import load_data_for_other_tabs, Basetab_data
from app import app


# app = dash.Dash(
#     __name__,
#     external_stylesheets=[dbc.themes.MORPH],
#     suppress_callback_exceptions=True,
# )

df_1 = Basetab_data  # load_data_for_other_tabs()  # pd.read_csv(


#     "/d/Solutions Team/Anomaly Project/Dash Tutorial/Dash2.0/eda_sample.csv"
# )
# df_1 = pd.DataFrame()\
def _get_tab1_layout(df_1):
    card_main = dbc.Card(
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
                        options=[i for i in df_1.columns],
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
                            html.Button(
                                "Reset Filter", id="reset-filter-val", n_clicks=0
                            ),
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
    tab1_layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(card_main, width=3),
                ],
                # justify="center",
            ),  # justify="start", "center", "end", "between", "around"
        ]
    )
    return tab1_layout


@app.callback(
    [
        Output("Filter-subsection-placeholder", "children"),
        Output("filter-dropdwn", "value"),
    ],
    [
        Input("add-filter-val", "n_clicks"),
        Input("reset-filter-val", "n_clicks"),
    ],
    State("filter-dropdwn", "value"),
    State("Filter-subsection-placeholder", "children"),
    prevent_initial_call=True,
)  # callback to update the subfilters in the selection panel
def update_selection_panel_sub_filter(
    n_clicks, reset_n_clicks, filter_value, filter_subsection_children
):
    subchildren = []
    if "add-filter-val" == ctx.triggered_id:
        existing_subfiters_id = [
            filter_subsection_children[i]["props"]["id"]
            for i in range(len(filter_subsection_children))
        ]
        for filter_col in filter_value:
            if filter_col + "_div" in existing_subfiters_id:
                subchildren.append(
                    filter_subsection_children[
                        existing_subfiters_id.index(filter_col + "_div")
                    ]
                )
            else:
                subchildren.append(
                    html.Div(
                        id=f"{filter_col}_div",
                        children=[
                            html.Br(),
                            html.Label(f"{filter_col}"),
                            dcc.Dropdown(
                                id={
                                    "type": "sub-filters-dropdown",
                                    "index": filter_value.index(
                                        filter_col
                                    ),  # temp_index,
                                    "col": filter_col,
                                    # "order" : temp_index
                                },
                                options=[val for val in df_1[f"{filter_col}"].unique()],
                                multi=True,
                                style={"color": "black"},
                            ),
                        ],
                    )
                )
        return subchildren, dash.no_update

    elif "reset-filter-val" == ctx.triggered_id:
        return subchildren, ""


@app.callback(
    Output({"type": "sub-filters-dropdown", "col": ALL, "index": MATCH}, "options"),
    [
        # Input({"type": "sub-filters-dropdown", "index": MATCH}, "value"),
        Input(
            {"type": "sub-filters-dropdown", "col": ALL, "index": ALLSMALLER}, "value"
        ),
    ],
    [
        # State({"type": "sub-filters-dropdown", "index": ALLSMALLER}, "value"),
        State({"type": "sub-filters-dropdown", "col": ALL, "index": ALL}, "id"),
        State({"type": "sub-filters-dropdown", "col": ALL, "index": MATCH}, "id"),
        State({"type": "sub-filters-dropdown", "col": ALL, "index": ALL}, "value"),
        # State("Filter-subsection-placeholder", "children"),
        State("filter-dropdwn", "value"),
    ],
    prevent_intial_call=True,  # won't work as the input used here is not available in the layout during the app start
)
def update_subfilters_dropdown(
    # matched_value,
    all_smaller,
    all_ids,
    matched_id,
    all_values,
    # filter_subsection_children,
    main_filter_value,
):
    # sub_filters_div_id_list = [i["props"]["id"] for i in filter_subsection_children]
    # index_filt = sub_filters_div_id_list.index(
    #     matched_id["index"] + "_div"
    # )  # replace this with filter dropdown list
    # index_filt = main_filter_value.index(matched_id["index"]) # main_filter_value.index(ctx.triggered_id['index'])

    if ctx.triggered_id:  # in main_filter_value:
        index_filt = main_filter_value.index(
            ctx.triggered_id["col"]
        )  # index of the col that is changed
        current_index = main_filter_value.index(
            matched_id[0]["col"]
        )  # index of the col that will updated by this callback
        if (
            current_index > index_filt
        ):  # now we are using ALLSMALLER, hence this check is not needed and we can remove the current_index
            df_2 = df_1  # .copy()
            for i in range(0, index_filt + 1):
                if all_ids[i] and all_values[i]:
                    df_2 = df_2[df_2[all_ids[i]["col"]].isin(all_values[i])]
            return (list(df_2[matched_id[0]["col"]].unique()),)
    return dash.no_update


def tab1_func_to_get_the_basetabdata(
    value,
):  # function to return the tab1 layout after the data got loaded in the datatable
    #print(value)
    global df_1
    df_1 = value
    # card_main = dbc.Card(
    #     [
    #         dbc.CardBody(
    #             [
    #                 html.H6("Selection Panel", className="card-title"),
    #                 html.P(
    #                     "Select Dimensions to Filter",
    #                     className="card-text",
    #                 ),
    #                 dcc.Dropdown(
    #                     id="filter-dropdwn",
    #                     options=[i for i in df_1.columns],
    #                     value="",
    #                     clearable=True,
    #                     multi=True,
    #                     style={"color": "#000000"},
    #                 ),
    #                 html.Br(),
    #                 html.Span(
    #                     [
    #                         html.Button("Add Filter", id="add-filter-val", n_clicks=0),
    #                         html.Spacer(),
    #                         html.Button(
    #                             "Reset Filter", id="reset-filter-val", n_clicks=0
    #                         ),
    #                     ]
    #                 ),
    #                 html.Br(),
    #                 html.Div(
    #                     id="Filter-subsection-placeholder",
    #                     children=[],
    #                 ),
    #                 html.Br(),
    #                 # dcc.Dropdown(
    #                 #     id="user_choice2",
    #                 #     options=["Input 1", "Input 2", "Input 3"],
    #                 #     value="Input 1",
    #                 #     clearable=False,
    #                 #     style={"color": "#000000"},
    #                 # ),
    #                 html.Button("Update Data", id="submit-val", n_clicks=0),
    #             ]
    #         ),
    #     ],
    #     color="dark",  # https://bootswatch.com/default/ for more card colors
    #     inverse=True,  # change color of text (black or white)
    #     outline=False,  # True = remove the block colors from the background and header
    # )

    # return dbc.Container(
    #     [
    #         dbc.Row(
    #             [
    #                 dbc.Col(card_main, width=3),
    #             ],
    #             # justify="center",
    #         ),  # justify="start", "center", "end", "between", "around"
    #     ]
    # )
    return _get_tab1_layout(df_1)
