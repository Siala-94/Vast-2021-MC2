from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
import datetime
import random
import colorsys
import numpy as np

# Define a function that generates a dictionary mapping each FullName to a unique color


def create_color_map(df):
    unique_names = df['FullName'].unique()
    N = len(unique_names)
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    hex_colors = [
        f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in RGB_tuples]
    random.shuffle(hex_colors)  # Shuffle the colors to create some variation

    color_map = dict(zip(unique_names, hex_colors))
    return color_map


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

    df_gps['date'] = df_gps["Timestamp"].dt.date
    df_gps['time'] = df_gps["Timestamp"].dt.time
    df_gps.drop("Timestamp", axis=1, inplace=True)

    return df_gps


def getHeatmap(df):
    df_count = df.groupby('location').size().reset_index(name='count')
    df_sorted = df_count.sort_values(by='count', ascending=False)

    fig = px.density_heatmap(
        df,
        x="date", y="location",
        color_continuous_scale='YlOrRd',
        nbinsx=len(df['date'].unique()),
        nbinsy=len(df['location'].unique()),
        category_orders={"location": df_sorted['location']})  # Set category order based on count

    # Reorder the locations based on the accumulated count
    fig.update_yaxes(categoryarray=df_sorted['location'])

    return fig


def getDates(df):
    return df['date'].unique()


def getLocations(df):
    return df['location'].unique()


def get4ccnum(df):
    return df['last4ccnum'].unique()


def getnames(df):
    return df['FullName'].unique()


def seconds_to_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def getsegments(map):
    segments = []

    for line in map.geometry:
        # Get the coordinates of the line
        coords = list(line.coords)

        # For each pair of points, create a separate line segment
        for i in range(len(coords) - 1):
            # Each segment is a dictionary with x, y and z (0) arrays of two points
            segment = dict(
                x=[coords[i][0], coords[i+1][0]],  # swap x and y
                y=[coords[i][1], coords[i+1][1]],
                z=[0, 0]
            )
            # Add the segment to the list
            segments.append(segment)
    return segments


def get3dscatter(df):

    return go.Scatter3d(
        x=df['long'],
        y=df['lat'],
        z=[0]*len(df),  # Add the markers on the ground (z=0)
        mode='markers',
        marker=dict(size=4, color='red'),
        # This adds the location names next to the markers
        text=df['location'],

        hoverinfo='text',
        hovertemplate="Longitude: %{x}<br>Latitude: %{y}<br>Location: %{text}<extra></extra>"
    )


def getGpsScatter(df, color_map, dim):
    # Calculate the time in seconds as before
    temp = df['time'].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)

    # Convert time to a formatted string
    time_str = df['time'].apply(lambda t: t.strftime('%H:%M:%S'))

    return go.Scatter3d(
        x=df['long'],
        y=df['lat'],
        z=[0]*len(df) if dim == '2D' else temp,
        mode='markers',
        marker=dict(
            size=3,
            # Color code by FullName
            color=df['FullName'].apply(lambda x: color_map[x]),
        ),
        hoverinfo='text',
        hovertemplate=(
            "Longitude: %{x}<br>"
            "Latitude: %{y}<br>"
            "Name: %{text}<br>"
            "Time: %{customdata[0]}<br>"
            "Employment Type: %{customdata[1]}<br>"
            "Employment Title: %{customdata[2]}<extra></extra>"
        ),
        text=df['FullName'],
        # Here we use customdata to pass time_str, CurrentEmploymentType, and CurrentEmploymentTitle to hovertemplate
        customdata=df[['time', 'CurrentEmploymentType',
                       'CurrentEmploymentTitle']].values.tolist(),
    )


# df = read_data("cc_data.csv", "loyalty_data.csv")
df = pd.read_csv("cards.csv")
df_gps = read_gps_data('gps.csv')
df_loc = pd.read_csv('location_coordinate.csv')
# Load map
abila_map = gpd.read_file('abila_clean.shp')
# Reproject to WGS84 (EPSG:4326)
abila_map = abila_map.to_crs("EPSG:4326")


app = Dash(__name__)

app.layout = html.Div([
    html.Div([

    ]),

    html.Div([
        # multi select
        html.Div([
            html.H6("choose a location to investigate",
                    style={'text-align': 'center', 'font-family': 'sans-serif'}),
            dcc.Dropdown(
                options=[loc for loc in getLocations(df)],
                id="multi",
                multi=True)
        ], style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
            html.H6("choose a creditcard to investigate",
                    style={'text-align': 'center', 'font-family': 'sans-serif'}),
            dcc.Dropdown(
                options=[num for num in getnames(df)],
                id="4cc",
                multi=True)
        ], style={'width': '50%', 'display': 'inline-block'})

    ]),

    # radio buttons
    html.Div([
        html.Div([
            html.H6("choose y-axis"),
            dcc.RadioItems(options=['time', 'location'],
                           id="y-axis",
                           value="time"
                           )
        ], style={'width': '20%', 'display': 'inline-block'}),
        html.Div([
            html.H6("choose x-axis"),
            dcc.RadioItems(options=['time', 'location'],
                           id="x-axis",
                           value="location"
                           )
        ], style={'width': '20%', 'display': 'inline-block'}),
        html.Div([
            html.H6("choose type of plot"),
            dcc.RadioItems(options=['Line Plot', 'Scatter Plot'],
                           id="type-plot",
                           value="Scatter Plot"
                           )
        ], style={'width': '20%', 'display': 'inline-block'}),
        html.Div([
            html.H6("choose size "),
            dcc.RadioItems(options=['Price', 'None'],
                           id="size",
                           value="None"
                           )
        ], style={'width': '20%', 'display': 'inline-block'}),
        html.Div([
            html.H6("choose cards "),
            dcc.RadioItems(options=['credit', 'loyaltynum'],
                           id="card",
                           value="credit"
                           )
        ], style={'width': '20%', 'display': 'inline-block'})
    ]),

    # graph and slider
    html.Div([
        html.Div([
            dcc.Graph(id="scatterplot")
        ], style={'width': '80%', 'height': '100%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id="heatmap")
        ], style={'width': '20%', 'display': 'inline-block'}),
        dcc.Slider(
            id='day-slider',
            # Subtract 1 to account for zero-based indexing
            max=len(getDates(df_gps)) - 1,
            min=0,
            value=len(getDates(df_gps)) - 1,
            marks={i: str(date.strftime("%b %d"))
                   for i, date in enumerate(getDates(df_gps))},
            step=1
        ),
        dcc.RangeSlider(
            id='time-slider',
            min=0,
            max=23,
            value=[0, 23],
            marks={i: f'{i}:00' for i in range(0, 24)},
            step=1
        ),

    ]),

    html.Div([
        dcc.Dropdown(options=[name for name in getnames(df_gps)],
                     id="names",
                     value=["Nils Calixto", "Loreto Bodrogi", "Minke Mies"],
                     multi=True),
        dcc.RadioItems(options=['2D', '3D'],
                       id="dimension",
                       value='2D'
                       ),
        dcc.Graph(id='scatter3d'),



    ])

])


@app.callback(
    Output('scatter3d', 'figure'),
    Input('day-slider', 'value'),
    Input('time-slider', 'value'),
    Input('names', 'value'),  # Add input for filtering by FullName
    Input('dimension', 'value')
)
def update_scatter3d(date, time_range, full_names, dim):  # Include full_names parameter

    # FILTER the data
    # filter based on date
    selected_date = getDates(df_gps)[date]
    df_filt = df_gps[df_gps['date'] == selected_date]
    # filter based on time
    # Convert the time_range values to time
    start_time = datetime.time(time_range[0])
    end_time = datetime.time(time_range[1])
    df_filt = df_filt[(df_filt['time'] >= start_time)
                      & (df_filt['time'] <= end_time)]
    # Filter the data based on selected full names
    df_filt = df_filt[df_filt['FullName'].isin(
        full_names)] if full_names else df_filt

    color_map = create_color_map(df_filt)
    fig = go.Figure()

    # Create a list to hold all the line segments
    segments = getsegments(abila_map)

    # Create scatter plot for each segment in the shapefile
    for segment in segments:
        shapefile_lines = go.Scatter3d(
            x=segment['x'],
            y=segment['y'],
            z=segment['z'],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        )
        fig.add_trace(shapefile_lines)

    # Create custom hover text
    hover_text = []
    for index, row in df_filt.iterrows():
        hover_text.append(
            f"FullName: {row['FullName']}, Time: {row['time'].strftime('%H:%M:%S')}")
    # Map FullName to numeric values using a categorical colormap
    unique_names = df_filt['FullName'].unique()
    # Create scatter plot for the GPS data
    gps_scatter = getGpsScatter(df_filt, color_map, dim)

    for name in unique_names:
        df_filt_name = df_filt[df_filt['FullName'] == name]
        gps_scatter = getGpsScatter(df_filt_name, color_map, dim)
        gps_scatter['name'] = name  # Set name for each trace
        fig.add_trace(gps_scatter)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='long'),
            yaxis=dict(title='lat'),
            zaxis=dict(title='time',
                       tickmode='array',
                       tickvals=list(range(0, 86401, 21600)),
                       ticktext=[seconds_to_time(i) for i in range(0, 86401, 21600)])  # convert seconds to time string
        ),
        height=700
    )

    # Add the new data points from the CSV file to the plot
    loc_scatter = get3dscatter(df_loc)

    fig.add_trace(loc_scatter)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='long'),
            yaxis=dict(title='lat'),
            zaxis=dict(title='time',
                       tickmode='array',
                       tickvals=list(range(0, 86401, 21600)),
                       ticktext=[seconds_to_time(i) for i in range(0, 86401, 21600)])  # convert seconds to time string
        ),
        height=700
    )

    return fig

# bar


@app.callback(
    Output('heatmap', 'figure'),
    Input('day-slider', 'value'),
    Input('multi', 'value')
)
def update_bar(date, locations):
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

# 2d plot


@app.callback(
    Output('scatterplot', 'figure'),
    Input('day-slider', 'value'),
    Input('multi', 'value'),
    Input('4cc', 'value'),
    Input('y-axis', 'value'),
    Input('x-axis', 'value'),
    Input('type-plot', 'value'),
    Input('size', 'value'),
    Input('card', 'value'),
    Input('time-slider', 'value')
)
def update_plot(date, locations, num, yaxis, xaxis, plot, size, card, time_range):
    selected_date = getDates(df)[date]
    # Make a copy of the filtered dataframe
    df_filt = df[df['date'] == selected_date].copy()

    # Filter by time range
    start_time = f"{time_range[0]:02d}:00:00"
    end_time = f"{time_range[1]:02d}:59:59"
    df_filt['time'] = pd.to_datetime(
        df_filt['time'])  # Convert 'time' to datetime
    df_filt = df_filt[(df_filt['time'].dt.strftime('%H:%M:%S') >= start_time) & (
        df_filt['time'].dt.strftime('%H:%M:%S') <= end_time)]

    # Filter by locations
    df_new = df_filt[df_filt['location'].isin(
        locations)] if locations else df_filt.copy()

    # Filter by credit card numbers
    df_new = df_new[df_new['FullName'].isin(num)] if num else df_new.copy()
    df_new['FullName'] = df_new['FullName'].astype(str)

    fig = None
    custom_data = df_new[['last4ccnum',
                          'loyaltynum', 'price', 'FullName']].values

    if plot == "Line Plot":
        fig = px.line(
            df_new,
            x=xaxis,
            y=yaxis,
            text="price",
            color="FullName" if card == 'credit' else "loyaltynum",
            color_discrete_sequence=px.colors.qualitative.Safe,
            category_orders={xaxis: sorted(df_new[xaxis].unique()),
                             'FullName': sorted(df_new['FullName'].unique())}
        )
    else:
        if size != 'Price':
            fig = px.scatter(
                df_new,
                x=xaxis,
                y=yaxis,
                color="FullName" if card == 'credit' else "loyaltynum",
                color_discrete_sequence=px.colors.qualitative.Safe,
                category_orders={'location': sorted(df_new['location'].unique()),
                                 'FullName': sorted(df_new['FullName'].unique())}
            )
        else:
            fig = px.scatter(
                df_new,
                x=xaxis,
                y=yaxis,
                size='price',
                color="FullName" if card == 'credit' else "loyaltynum",
                color_discrete_sequence=px.colors.qualitative.Safe,
                category_orders={'location': sorted(df_new['location'].unique()),
                                 'FullName': sorted(df_new['FullName'].unique())}
            )

    fig.update_layout(
        transition_duration=500,
        hoverlabel=dict(
            bgcolor="white",  # Set the background color of the hover label
            font_size=12,  # Set the font size
            font_family="Arial"  # Set the font family of the hover label text
        )
    )

    fig.update_traces(
        mode='markers+lines',
        marker={'sizemode': 'area'})

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
