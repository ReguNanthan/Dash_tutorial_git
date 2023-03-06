import dash
import dash_table
import pandas as pd
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import html
from DataLoader import Basetab_data
from app import app

# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# df = pd.read_csv("Dash Tutorial/Dash2.0/County.csv")

df = Basetab_data


def _get_tab2_layout(df):
    if not df.empty:
        table_1 = dash_table.DataTable(
            id="table_1",
            columns=[{"name": i, "id": i} for i in ["County", "Sales_Amt"]],
            data=df.groupby(["County"])
            .aggregate("sum")
            .reset_index()
            .to_dict("records"),
        )

        table_2 = dash_table.DataTable(
            id="table_2",
            columns=[{"name": i, "id": i} for i in ["Base_UPC", "Sales_Amt"]],
            data=df[["Base_UPC", "Sales_Amt"]].to_dict("records"),
        )
    else:
        table_1, table_2 = dash_table.DataTable(
            id="table_1",
        ), dash_table.DataTable(
            id="table_2",
        )

    modal = html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Filtered Data")),
                    dbc.ModalBody(
                        table_2,
                    ),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close", className="ms-auto", n_clicks=0)
                    ),
                ],
                id="modal",
                size="lg",
                is_open=False,
                scrollable=True,
            ),
        ]
    )

    tab2_layout = html.Div(
        [
            table_1,
            modal,
        ]
    )
    return tab2_layout


# if not df.empty:
#     table_1 = dash_table.DataTable(
#         id="table_1",
#         columns=[{"name": i, "id": i} for i in ["County", "Sales_Amt"]],
#         data=df.groupby(["County"]).aggregate("sum").reset_index().to_dict("records"),
#     )

#     table_2 = dash_table.DataTable(
#         id="table_2",
#         columns=[{"name": i, "id": i} for i in ["Base_UPC", "Sales_Amt"]],
#         data=df[["Base_UPC", "Sales_Amt"]].to_dict("records"),
#     )
# else:
#     table_1, table_2 = dash_table.DataTable(
#         id="table_1",
#     ), dash_table.DataTable(
#         id="table_2",
#     )


@app.callback(
    Output("table_2", "data"),
    [Input("table_1", "active_cell")],
    [State("table_1", "data"), State("table_2", "data")],
)
def update_state_table(active_cell, country_table_data, state_table_data):
    if active_cell:
        filter_col = list(country_table_data[active_cell["row"]].keys())[0]
        filter_val = list(country_table_data[active_cell["row"]].values())[0]
        cols_needed_for_output = ["Base_UPC", "Sales_Amt"] #list(state_table_data[0].keys())
        # County = country_table_data[active_cell['row']]['County'] #list(country_table_data[active_cell['row']].values())[0]
        # filtered_df = df[df['County'] == County][['Base_UPC', 'Sales_Amt']]
        filtered_df = df[df[filter_col] == filter_val][cols_needed_for_output]

        return filtered_df.to_dict("records")
    # else:
    #     return df[["Base_UPC", "Sales_Amt"]].to_dict("records")


@app.callback(
    Output("table_1", "style_data_conditional"),
    Input("table_1", "active_cell"),
    State("table_1", "style_data_conditional"),
)
def highlight_selected_row(active_cell, style_data_conditional):
    if active_cell:
        highlight_row = active_cell["row"]
        style_data_conditional = [
            {
                "if": {
                    "row_index": highlight_row,
                },
                "backgroundColor": "dodgerblue",
                "color": "white",
            },
            {
                "if": {
                    "state": "active",
                },
                "backgroundColor": "dodgerblue",
                "color": "white",
            },
        ]
    return style_data_conditional


@app.callback(
    Output("modal", "is_open"),
    [Input("close", "n_clicks"), Input("table_1", "active_cell")],
    [State("modal", "is_open")],
)
def toggle_modal(close_modal_button, active_cell, is_open):
    if active_cell or close_modal_button:
        return not is_open
    return is_open


def tab2_func_to_get_the_basetabdata(Basetab_data):
    global df
    df = Basetab_data
    return _get_tab2_layout(Basetab_data)


tab2_layout = _get_tab2_layout(Basetab_data)
