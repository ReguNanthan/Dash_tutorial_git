import dash
from dash import html
import dash_bootstrap_components as dbc
from dash import dcc, dash_table, Dash, ctx
from dash.dependencies import Output, Input, State, MATCH, ALL, ALLSMALLER
import os
import pandas as pd
import base64
import io

import yaml
from app import app


from _cloud_utils.azure.adls import ADLSUtil


# app = dash.Dash(
#     __name__,
#     external_stylesheets=[dbc.themes.MORPH],
#     suppress_callback_exceptions=True,
# )

Basetab_data = pd.DataFrame()  # global df

button_group = html.Div(
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
    style={"formCheck ": "padding-left: 0"}
    # style={
    #     "padding-left": "0",
    #     # ".btn-group .form-check": {"padding-left": "0"},
    #     # ".btn-group .btn-group > .form-check:not(:last-child) > .btn": {
    #     #     "border-top-right-radius": "0",
    #     #     "border-bottom-right-radius": "0",
    #     # },
    #     # ".btn-group .btn-group > .form-check:not(:first-child) > .btn": {
    #     #     "border-top-left-radius": "0",
    #     #     "border-bottom-left-radius": "0",
    #     #     "margin-left": "-1px",
    #     # },
    # },
    # style={
    #     ".radio-group .form-check": {"padding-left": "0"},
    #     ".radio-group .btn-group>.form-check:not(:last-child)>.btn": {
    #         "border-top-right-radius": "0",
    #         "border-bottom-right-radius": "0",
    #     },
    #     ".radio-group .btn-group>.form-check:not(:first-child)>.btn": {
    #         "border-top-left-radius": "0",
    #         "border-bottom-left-radius": "0",
    #         "margin-left": "-1px",
    #     },
    # },
)


dataloader_layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H4("Data Loader"),
                width={"size": 8, "offset": 3},
            ),
            justify="center",
        ),
        html.Br(),
        dbc.Row(
            dbc.Col(button_group, width={"offset": 3, "size": 6}),
        ),
        html.Br(),
        dbc.Row(dbc.Col(html.Div(id="Upload_place_holder"))),
        html.Br(),
        dbc.Row(
            dbc.Col(
                html.Button("Load Data", id="load-data", n_clicks=0),
                width={"size": 10, "offset": 3},
            )
        ),
        # html.Div(
        #     id="output-data-message-placeholder"
        # ),  # component to ensure file is loaded #testing
    ],
    style={"border": "3px solid black", "padding": "10px", "margin": "10px"},
    # fluid=True,
)


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif "parquet" in filename:
            # Assume that the user uploaded a parquet file
            df = pd.read_parquet(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

    return df, [
        html.Div(
            [
                html.H5(filename),
                # html.H6(datetime.datetime.fromtimestamp(date)),
                dash_table.DataTable(
                    data=df.to_dict("records"),
                    columns=[{"name": i, "id": i} for i in df.columns],
                ),
                html.Hr(),  # horizontal line
            ]
        )
    ]


@app.callback(
    Output("Upload_place_holder", "children"), Input("radios", "value")
)  # Based on the upload options
def create_upload_div(upload_option):  # provide the upload facility
    if upload_option == "Local":
        return html.Div(
            [
                html.Br(),
                dcc.Upload(
                    id="upload-data-local",
                    children=html.Div(
                        [
                            html.Div(id="file-name"),
                            "Drag and Drop or ",
                            html.A("Select Files"),
                        ]
                    ),
                    style={
                        "width": "70%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px",
                    },
                    # Allow multiple files to be uploaded
                    multiple=False,
                ),
                html.Div(id="output-data-message-placeholder"),
            ]
        )
    elif upload_option == "data/raw":
        return html.Div(
            [
                html.H6("Choose file from raw directory"),
                dcc.Dropdown(
                    id="upload-data-raw",  # raw-dropdown
                    options=[
                        {"label": option, "value": option}
                        for option in os.listdir("Dash_App_with_Tabs/data/raw")
                    ],
                    value="Dash_tutorial_git/Dash_App_with_Tabs/data/raw",
                    multi=False,
                    style={
                        "width": "100%",
                    },
                ),
                html.Div(id="output-data-message-placeholder2"),
            ]
        )
    elif upload_option == "Cloud":
        return html.Div(
            [
                html.H6("Choose cloud platform"),
                dcc.Dropdown(
                    id="upload-data-options-cloud",  # cloud options
                    options=["Azure", "AWS", "GCP"],
                    value="Azure",
                    multi=False,
                    style={
                        "width": "50%",
                    },
                ),
                html.Div(id="output-data-message-placeholder3"),
            ]
        )


@app.callback(  # callback to get the data from raw input file
    Output("output-data-message-placeholder2", "children"),
    Input("load-data", "n_clicks"),
    State("upload-data-raw", "value"),
    State("radios", "value"),
    prevent_initial_call=True,
)
def update_output_raw(
    load_data_click,
    file_name,
    upload_option,
):
    if load_data_click:
        if upload_option == "data/raw":
            if file_name is not None:
                file_path = os.path.join("Dash_App_with_Tabs/data/raw", file_name)
                global Basetab_data
                Basetab_data = pd.read_csv(file_path)
                return html.Div(
                    [
                        html.Br(),
                        html.H5(f"{file_name} uploaded successfully"),
                        html.Br(),
                    ]
                )


@app.callback(  # callback to indicate the filename once we click the file in Local
    [Output("upload-data-local", "children")],
    [Input("upload-data-local", "filename")],
    prevent_initial_call=True,
)
def update_filename_local(filename):
    if filename is not None:
        return [html.Div(filename)]
    # else:
    #     return [
    #         html.Div(
    #             [
    #                 html.Div(id="file-name"),
    #                 "Drag and Drop or ",
    #                 html.A("Select Files"),
    #             ]
    #         )
    #     ]
    #    return []
    # return html.Div("No file uploaded")


@app.callback(  # callback to get the data from the upload file in Local
    Output("output-data-message-placeholder", "children"),
    Input("load-data", "n_clicks"),
    State("radios", "value"),
    State("upload-data-local", "contents"),
    State("upload-data-local", "filename"),
    State("upload-data-local", "last_modified"),
    prevent_initial_call=True,
)
def update_output_local(
    load_data_click,
    upload_option,
    list_of_contents,
    list_of_names,
    list_of_dates,
):
    if load_data_click:
        if upload_option == "Local" and list_of_contents is not None:
            global Basetab_data
            Basetab_data, children = parse_contents(
                list_of_contents, list_of_names, list_of_dates
            )
            return children


# Get the details from config file
def _get_config_dict(
    filename,
):
    with open(filename, "r") as stream:
        # try:
        config = yaml.safe_load(stream)
        return config
    # except yaml.YAMLError as exc:


config = _get_config_dict(
    filename="/d/Solutions Team/Anomaly Project/Dash_tutorial_git/Dash_App_with_Tabs/config/config.yml"
)


# Get the data from cloud
def _get_data_from_cloud(cloud_name):
    if cloud_name == "Azure":
        azure_config_details = config.get("Azure")
        adls_obj = ADLSUtil(azure_config_details)
        return adls_obj.read_process_adls_blobs()  # data


# export the data to cloud
def _export_data_to_cloud(cloud_name, export_df):
    if cloud_name == "Azure":
        azure_config_details = config.get("Azure")
        adls_obj = ADLSUtil(azure_config_details)
        adls_obj.write_to_adls(
            df=export_df,
            exe=azure_config_details["export_ext"],
        )


@app.callback(
    Output("output-data-message-placeholder3", "children"),
    [
        # Input("upload-data-options-cloud", "value"),
        Input("load-data", "n_clicks"),
    ],
    [
        State("upload-data-options-cloud", "value"),
        State("radios", "value"),
    ],
    prevent_initial_call=True,
)
def get_data_from_cloud(
    n_clicks,
    cloud_name,
    data_load_option,
):
    if data_load_option == "Cloud" and cloud_name == "Azure":
        global Basetab_data
        Basetab_data = _get_data_from_cloud(cloud_name)

        # _export_data_to_cloud("Azure", Basetab_data)

        return html.Div(
            [
                # html.H5(filename),
                # html.H6(datetime.datetime.fromtimestamp(date)),
                dash_table.DataTable(
                    data=Basetab_data.to_dict("records"),
                    columns=[{"name": i, "id": i} for i in Basetab_data.columns],
                ),
                html.Hr(),  # horizontal line
            ]
        )


def load_data_for_other_tabs():
    # global Basetab_data
    return Basetab_data


# if __name__ == "__main__":
#     app.layout = dataloader_layout
#     app.run_server(debug=False)
