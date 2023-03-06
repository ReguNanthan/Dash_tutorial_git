import dash
from dash import html
import dash_bootstrap_components as dbc

# from dash import dcc, dash_table, Dash, ctx
# from dash.dependencies import Output, Input, State, MATCH, ALL, ALLSMALLER
# import dash

# import pyarrow, os
# import pandas as pd

# import base64
# import io
# from app import app
# from DataLoader import dataloader_layout
# from Tab1 import tab1_layout

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    # meta_tags=[
    #     {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    # ],
)
