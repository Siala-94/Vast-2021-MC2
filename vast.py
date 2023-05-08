# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# Incorporate data
df = pd.read_csv('./MC2/gps.csv')

# Initialize the app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.Div(children='Space-Time Cube Scatterplot'),
    dcc.Graph(id='scatterplot')
])

# Callback to create scatterplot


@app.callback(
    Output(component_id='scatterplot', component_property='figure'),
    Input(component_id='scatterplot', component_property='n_clicks')
)
def update_scatterplot(n_clicks):
    figure = px.scatter_3d(df, x='lat', y='long', z='Timestamp',
                           color='Timestamp', hover_name='Timestamp',
                           labels={'lat': 'Latitude', 'long': 'Longitude', 'Timestamp': 'Time'})
    return figure


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
