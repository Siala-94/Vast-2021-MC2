from dash import Dash, html, dcc, Input, Output

import pandas as pd
import plotly.graph_objs as go
from datetime import datetime

# Step 2: Read in the data file and preprocess it
creditCardData = pd.read_csv('MC2/cc_data.csv', encoding='latin-1')
creditCardData['timestamp'] = pd.to_datetime(creditCardData['timestamp'])
creditCardData['timestamp'] = creditCardData['timestamp'].dt.date
cc_counts = creditCardData.groupby(
    ['location', 'timestamp']).size().reset_index(name='count')
cc_heatmap_data = cc_counts.pivot(
    index='location', columns='timestamp', values='count')
cc_heatmap_data = cc_heatmap_data.fillna(0)  # fill NaN values with 0

# Step 3: Create a new Dash application
app = Dash(__name__)

# Step 4: Define the layout of our Dash application
app.layout = html.Div([
    html.H1('Popular Locations and Dates for Credit Card Data'),
    dcc.Dropdown(
        id='location-dropdown',
        options=[{'label': 'All', 'value': 'all'}] +
        [{'label': loc, 'value': loc} for loc in cc_heatmap_data.index],
        value='all'
    ),
    dcc.Graph(id='heatmap')
])

# Step 5: Define a callback function that updates the heatmap


@app.callback(
    Output('heatmap', 'figure'),
    Input('location-dropdown', 'value')
)
def update_heatmap(location):
    if location == 'all':
        data = go.Heatmap(
            z=cc_heatmap_data.values.tolist(),
            x=cc_heatmap_data.columns.tolist(),
            y=cc_heatmap_data.index.tolist(),
            colorscale='Blues'
        )
    else:
        data = go.Heatmap(
            z=cc_heatmap_data.loc[location].values.tolist(),
            x=cc_heatmap_data.columns.tolist(),
            y=[location],
            colorscale='Blues'
        )
    layout = go.Layout(
        title='Credit Card Transactions',
        xaxis={'title': 'Date'},
        yaxis={'title': 'Location'}
    )
    return go.Figure(data=[data], layout=layout)


# Step 6: Run the Dash application
if __name__ == '__main__':
    app.run_server(debug=True)
