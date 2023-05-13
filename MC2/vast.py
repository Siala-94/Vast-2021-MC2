# Import necessary libraries
from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime

# Load data from csv files
cc_data = pd.read_csv('cc_data.csv', encoding="latin-1")
loyalty_data = pd.read_csv('loyalty_data.csv', encoding="latin-1")

# Convert timestamp columns to date format
cc_data['timestamp'] = pd.to_datetime(cc_data['timestamp']).dt.date
loyalty_data['timestamp'] = pd.to_datetime(loyalty_data['timestamp']).dt.date

# Group and aggregate data
cc_counts = cc_data.groupby(
    ['location', 'timestamp']).size().reset_index(name='count')
loyalty_counts = loyalty_data.groupby(
    ['location', 'timestamp']).size().reset_index(name='count')

# Pivot data for heatmap
cc_heatmap_data = cc_counts.pivot(
    index='location', columns='timestamp', values='count')
loyalty_heatmap_data = loyalty_counts.pivot(
    index='location', columns='timestamp', values='count')

# Fill missing values with zeros
cc_heatmap_data = cc_heatmap_data.fillna(0)
loyalty_heatmap_data = loyalty_heatmap_data.fillna(0)

# Combine data from credit card and loyalty transactions
combined_heatmap_data = cc_heatmap_data.add(loyalty_heatmap_data, fill_value=0)
combined_heatmap_data = combined_heatmap_data.fillna(0)

# Create a new Dash application
app = Dash(__name__)

# Define the layout of the Dash application
app.layout = html.Div([
    html.Div([
        html.H1('Popular Locations and Dates for Credit Card Data'),
        dcc.Graph(id='heatmap')
    ], style={'width': '50%', 'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(
            id='location-dropdown',
            options=[{'label': 'All', 'value': 'all'}] +
            [{'label': loc, 'value': loc} for loc in cc_heatmap_data.index],
            value='all'
        ),
        dcc.Graph(id='polar-plot'),
        dcc.Slider(
            id='day-slider',
            min=0,
            max=len(combined_heatmap_data.columns) - 1,
            value=len(combined_heatmap_data.columns) - 1,
            marks={i: str(date.strftime("%b %d"))
                   for i, date in enumerate(combined_heatmap_data.columns)},
            step=1
        )
    ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'})
])


# Define a callback function that updates the heatmap
@app.callback(
    Output('heatmap', 'figure'),
    Input('location-dropdown', 'value')
)
def update_heatmap(location):
    if location == 'all':
        data = go.Heatmap(
            z=combined_heatmap_data.values.tolist(),
            x=combined_heatmap_data.columns.tolist(),
            y=combined_heatmap_data.index.tolist(),
            colorscale='ylorrd'
        )
    else:
        if location in combined_heatmap_data.index:
            data = go.Heatmap(
                z=combined_heatmap_data.loc[location].values.tolist(),
                x=combined_heatmap_data.columns.tolist(),
                y=[location],
                colorscale='ylorrd'
            )
        else:
            data = go.Heatmap(
                z=[[0]],
                x=[combined_heatmap_data.columns[0]],
                y=[location],
                colorscale='ylorrd'
            )
    layout = go.Layout(
        title='Credit Card and Loyalty Transactions',
        # Increase the font size
        xaxis={'title': 'Date', 'tickfont': {'size': 10}},
        yaxis={'title': 'Location', 'tickfont': {
            'size': 10}}  # Increase the font size
    )
    return go.Figure(data=[data], layout=layout)


@app.callback(
    Output('polar-plot', 'figure'),
    Input('location-dropdown', 'value'),
    Input('day-slider', 'value')
)
def update_polar_plot(location, day):
    selected_day = combined_heatmap_data.columns[day]

    if location == 'all':
        data = []
        for loc in cc_heatmap_data.index:
            transactions = cc_data[(cc_data['location'] == loc) & (
                cc_data['timestamp'] == selected_day)].copy()
            transactions['timestamp'] = pd.to_datetime(
                transactions['timestamp'])
            hour_counts = transactions.groupby(
                transactions['timestamp'].dt.hour).size().reset_index(name='count')
            angles = [i * 15 for i in hour_counts['timestamp']]
            hour_counts = hour_counts['count'].tolist()
            data.append(go.Scatterpolar(
                r=hour_counts,
                theta=angles,
                fill='toself',
                name=loc,
                mode='markers+lines',
                marker=dict(size=5)
            ))
    else:
        if location in cc_heatmap_data.index:
            transactions = cc_data[(cc_data['location'] == location) & (
                cc_data['timestamp'] == selected_day)].copy()
            transactions['timestamp'] = pd.to_datetime(
                transactions['timestamp'])
            hour_counts = transactions.groupby(
                transactions['timestamp'].dt.hour).size().reset_index(name='count')
            angles = [i * 15 for i in hour_counts['timestamp']]
            hour_counts = hour_counts['count'].tolist()
            data = [go.Scatterpolar(
                r=hour_counts,
                theta=angles,
                fill='toself',
                name=location,
                mode='markers+lines',
                marker=dict(size=5)
            )]
        else:
            data = []

    layout = go.Layout(
        title='Transactions over the Day',
        polar=dict(
            radialaxis=dict(
                tickvals=[0, 10, 20, 30, 40, 50, 60],
                ticktext=['0', '10', '20', '30', '40', '50', '60']
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=[i * 15 for i in range(24)],
                ticktext=[str(i) + ':00' for i in range(24)]
            )
        )
    )
    return go.Figure(data=data, layout=layout)


# Step 7: Run the Dash application
if __name__ == '__main__':
    app.run_server(debug=True)
