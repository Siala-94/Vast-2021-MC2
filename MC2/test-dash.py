from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os


def read_data(data1, data2):
    cc_data = pd.read_csv(data1, encoding="latin-1")
    loyalty_data = pd.read_csv(data2, encoding="latin-1")

    # splitting date and time for credit
    cc_data[['date', 'time']] = cc_data['timestamp'].str.split(
        ' ', n=1, expand=True)
    cc_data.drop('timestamp', axis=1, inplace=True)
    loyalty_data.rename(columns={'timestamp': 'date'}, inplace=True)

    # adding card type
    cc_data['cardtype'] = 'credit'
    loyalty_data['cardtype'] = 'loyalty'

    # concatenating the dataframes
    df = pd.concat([cc_data, loyalty_data], ignore_index=True)
    # Convert to datetime if not already in datetime format
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day_name()
    df['last4ccnum'] = df['last4ccnum'].astype(str)

    return df


def read_gps_data(data):

    # Read the GPS data from the CSV file
    df_gps = pd.read_csv('gps.csv')
    df_gps['Timestamp'] = pd.to_datetime(
        df_gps['Timestamp'], format='%m/%d/%Y %H:%M:%S')
    df_gps.sort_values(by='Timestamp', inplace=True)

    # Read the car assignment data from the CSV file
    df_car = pd.read_csv('car-assignments.csv')

    # Merge the data frames based on the "id" column
    df_merged = pd.merge(df_gps, df_car, left_on='id', right_on='CarID')

    df_merged.drop("CarID", axis=1, inplace=True)

    df_merged['date'] = df_merged["Timestamp"].dt.date
    df_merged['time'] = df_merged["Timestamp"].dt.time
    df_merged.drop("Timestamp", axis=1, inplace=True)

    return df_merged


def getHeatmap(df):
    df_sorted = df.sort_values(by='date')

    fig = px.density_heatmap(
        df_sorted,
        x="date", y="location",
        color_continuous_scale='ylorrd',
        nbinsx=len(df['date'].unique()),
        nbinsy=len(df['location'].unique()))

    return fig


def getDates(df):
    return df['date'].unique()


def getLocations(df):
    return df['location'].unique()


def get4ccnum(df):
    return df['last4ccnum'].unique()


df = read_data("cc_data.csv", "loyalty_data.csv")

df_gps = read_gps_data('gps.csv')

app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(figure=getHeatmap(df))
    ]),

    html.Div([
        # multi select
        html.Div([
            html.H6("choose a location to investigate", style={
                'text-align': 'center', 'font-family': 'sans-serif'}),
            dcc.Dropdown(options=[{'label': loc, 'value': loc}
                                  for loc in getLocations(df)], id="multi", multi=True)
        ], style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
            html.H6("choose a creditcard to investigate", style={
                'text-align': 'center', 'font-family': 'sans-serif'}),
            dcc.Dropdown(options=[{'label': loc, 'value': loc}
                                  for loc in get4ccnum(df)], id="4cc", multi=True)
        ], style={'width': '50%', 'display': 'inline-block'})

    ]),

    # radio buttons
    html.Div([
        html.Div([
            html.H6("choose y-axis"),
            dcc.RadioItems(options=[{'label': 'Time', 'value': 'time'},
                                    {'label': 'Location', 'value': 'location'}],
                           id="y-axis",
                           style={'display': 'inline-block'})
        ], style={'width': '33%', 'display': 'inline-block'}),
        html.Div([
            html.H6("choose x-axis"),
            dcc.RadioItems(options=[{'label': 'Time', 'value': 'time'},
                                    {'label': 'Location', 'value': 'location'}],
                           id="x-axis",
                           style={'display': 'inline-block'})
        ], style={'width': '33%', 'display': 'inline-block'}),
        html.Div([
            html.H6("choose type of plot"),
            dcc.RadioItems(options=[{'label': 'Line Plot', 'value': 'line'},
                                    {'label': 'Scatter Plot', 'value': 'scatter'}],
                           id="type-plot",
                           style={'display': 'inline-block'})
        ], style={'width': '33%', 'display': 'inline-block'})
    ]),

    # graph and slider
    html.Div([
        html.Div([
            dcc.Graph(id="scatterplot")
        ], style={'width': '80%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id="heatmap")
        ], style={'width': '20%', 'display': 'inline-block'}),
        dcc.Slider(
            id='day-slider',
            max=len(getDates(df))-1,
            min=0,
            value=len(getDates(df))-1,
            marks={i: str(date.strftime("%b %d"))
                   for i, date in enumerate(getDates(df))},
            step=1
        )
    ]),

    html.Div([
        dcc.RadioItems(options=[{'label': '3d line Plot', 'value': 'line'},
                                {'label': '3d Scatter Plot', 'value': 'scatter'}],
                       id="3dplot"),
        dcc.Graph(id='line3d')

    ])

])


@app.callback(
    Output('line3d', 'figure'),
    Input('3dplot', 'value')
)
def update_3d(plot):
    selected_date = getDates(df_gps)[-1]  # Use the last date by default
    df_filt = df_gps[df_gps['date'] == selected_date]

    fig = None
    if plot == 'line':
        fig = px.line_3d(df_filt, x="long", y="lat",
                         z='time', color="FirstName")
    else:
        fig = px.scatter_3d(df_filt, x="long", y="lat",
                            z="time", color="FirstName")
        fig.update_traces(marker=dict(size=2))
    return fig


@app.callback(
    Output('scatter3d', 'figure'),
    Input('day-slider', 'value')
)
def update_scatter3d(date):
    selected_date = getDates(df_gps)[date]
    df_filt = df_gps[df_gps['date'] == selected_date]

    fig = px.scatter_3d(df_filt, x="long", y="lat",
                        z="time", color="FirstName")

    fig.update_traces(marker=dict(size=2))

    return fig


@app.callback(
    Output('heatmap', 'figure'),
    Input('day-slider', 'value'),
    Input('multi', 'value')
)
def update_heatmap(date, locations):
    selected_date = getDates(df)[date]

    df_filt = df[df['date'] == selected_date]

    df_filtered = df_filt[df_filt['location'].isin(
        locations)] if locations else df_filt

    df_count = df_filtered.groupby('location').size().reset_index(name='count')
    df_sorted = df_count.sort_values(by='count', ascending=True)

    fig = px.bar(
        df_sorted,
        x='count',
        y='location',
        color='count',
        color_continuous_scale='YlOrRd',
        orientation='h'
    )

    fig.update_layout(
        yaxis={'title': 'Locations'},
        xaxis={'title': 'Count'}
    )

    return fig


@app.callback(
    Output('scatterplot', 'figure'),
    Input('day-slider', 'value'),
    Input('multi', 'value'),
    Input('4cc', 'value'),
    Input('y-axis', 'value'),
    Input('x-axis', 'value'),
    Input('type-plot', 'value')

)
def update_plot(date, locations, num, yaxis, xaxis, plot):
    selected_date = getDates(df)[date]

    df_filt = df[df['date'] == selected_date]
    df_new = df_filt[df_filt['location'].isin(
        locations)] if locations else df_filt

    df_new = df_new[df_new['last4ccnum'].isin(num)] if num else df_new

    # Convert time to datetime data type
    df_new['time'] = pd.to_datetime(df_new['time'])
    df_new = df_new.sort_values(by='time')  # Sort by time

    fig = None

    if plot == "line":
        fig = px.line(
            df_new,
            x=xaxis,
            y=yaxis,
            text="price",
            color="last4ccnum",
            color_discrete_sequence=px.colors.qualitative.Safe,  # Set a fixed color sequence
            category_orders={xaxis: sorted(df_new[xaxis].unique()),
                             'last4ccnum': sorted(df_new['last4ccnum'].unique())}  # Set category orders for x-axis and last4ccnum
        )
    else:
        fig = px.scatter(
            df_new,
            x=xaxis,
            y=yaxis,
            size="price",
            color="last4ccnum",
            color_discrete_sequence=px.colors.qualitative.Safe,  # Set a fixed color sequence
            category_orders={'location': sorted(df_new['location'].unique()),
                             'last4ccnum': sorted(df_new['last4ccnum'].unique())}  # Set category orders for location and last4ccnum
        )

    fig.update_layout(transition_duration=500)

    return fig


# ...


# ...
if __name__ == '__main__':
    app.run_server(debug=True)
