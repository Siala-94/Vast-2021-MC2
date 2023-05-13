from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


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


def getHeatmap(df):
    df_sorted = df.sort_values(by='date')
    fig = px.density_heatmap(
        df_sorted, x="date", y="location", color_continuous_scale='ylorrd')
    fig.update_layout(xaxis={'range': [df['date'].min(), df['date'].max()]})
    return fig


def getDates(df):
    return df['date'].unique()


def getLocations(df):
    return df['location'].unique()


def get4ccnum(df):
    return df['last4ccnum'].unique()


df = read_data("cc_data.csv", "loyalty_data.csv")


app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(figure=getHeatmap(df))
    ]),

    html.Div([
        html.Div([
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

        html.Div([
            dcc.Graph(id="scatterplot"),
            dcc.Slider(
                id='day-slider',
                max=len(getDates(df))-1,
                min=0,
                value=len(getDates(df))-1,
                marks={i: str(date.strftime("%b %d"))
                       for i, date in enumerate(getDates(df))},
                step=1
            )
        ])

    ])

])


@app.callback(
    Output('scatterplot', 'figure'),
    Input('day-slider', 'value'),
    Input('multi', 'value'),
    Input('4cc', 'value')
)
def update_plot(date, locations, num):
    selected_date = getDates(df)[date]

    df_new = df[df['date'] == selected_date]

    df_new = df_new[df_new['location'].isin(
        locations)] if locations else df_new

    df_new = df_new[df_new['last4ccnum'].isin(num)] if num else df_new

    fig = px.scatter(df_new, x="time", y="location", color="last4ccnum")
    fig.update_layout(transition_duration=500)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
