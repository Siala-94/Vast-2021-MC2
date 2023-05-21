import pandas as pd
import plotly.graph_objs as go
from dash import Dash, html, dcc, Input, Output
import geopandas as gpd

# Load GPS data from CSV
df = pd.read_csv('gps.csv')[:1000]
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Load shapefile for geographical mapping
gdf = gpd.read_file('Geospatial/Kronos_Island.shp')

# Create Dash application
app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='gps-movement'),
    dcc.Interval(
        id='interval-component',
        interval=1000,  # Update every second
        n_intervals=0
    )
])


@app.callback(
    Output('gps-movement', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_gps_movement(n):
    traces = []
    for i in range(min(1000, len(df))):
        trace = go.Scattermapbox(
            lat=[df['lat'][i]],
            lon=[df['long'][i]],
            mode='markers',
            marker=dict(
                size=4,
                color='blue',
                opacity=0.8
            ),
            name=df['id'][i]
        )
        traces.append(trace)

    layout = go.Layout(
        title='GPS Movement',
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=gdf.centroid.y, lon=gdf.centroid.x),
            zoom=10
        ),
        autosize=True
    )

    return {'data': traces, 'layout': layout}


if __name__ == '__main__':
    app.run_server(debug=True)
