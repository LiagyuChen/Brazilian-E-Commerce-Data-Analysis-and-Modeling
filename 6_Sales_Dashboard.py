import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go

# Create a dash application
app = dash.Dash(__name__)

app.config.suppress_callback_exceptions = True

# Read the data into pandas dataframe
df = pd.read_csv('data/brazilian_sell.csv')

# Application layout
app.layout = html.Div(children=[
    # Add title to the dashboard
    html.H1(
        children='Brazilian Dataset Sales Dashboard',
        style={
            'textAlign': 'center',
            'color': '#503D36',
            'fontSize': 40
        }
    ),
    # Add a dropdown bar to choose a year
    html.Div([
        html.Div([
            html.Div([html.H2('Choose Year:', style={'margin-right': '2em'})]),
            dcc.Dropdown(id='input-year',
                         options=[{'label': i, 'value': i} for i in [2016, 2017, 2018]],
                         placeholder="Select a year",
                         style={'width': '85%', 'padding-left': '5px', 'font-size': '20px', 'text-align-last': 'center'}),
        ], style={'display': 'flex'}),
    ]),
    # Empty divs for the graphs
    html.Div([html.Div([], id='graph1', style={'width': '85%'}), html.Div([], id='graph2', style={'width': '85%'}), html.Div([], id='graph3', style={'width': '85%'})], style={'display': 'flex'}),
    html.Div([html.Div([], id='graph4', style={'width': '80%'}), html.Div([], id='graph5', style={'width': '80%'}), html.Div([], id='graph6', style={'width': '95%'})], style={'display': 'flex'}),
    html.Div([html.Div([], id='graph7', style={'width': '85%'}), html.Div([], id='graph8', style={'width': '85%'}), html.Div([], id='graph9', style={'width': '85%'})], style={'display': 'flex'}),
    html.Div([html.Div([], id='graph10', style={'width': '90%'}), html.Div([], id='graph11', style={'width': '90%'}), html.Div([], id='graph12', style={'width': '80%'})], style={'display': 'flex'}),
    html.Div([html.Div([], id='graph13', style={'width': '45%'}), html.Div([], id='graph14', style={'width': '120%'})], style={'display': 'flex'}),
    html.Div([html.Div([], id='graph15', style={'width': '85%'}), html.Div([], id='graph16', style={'width': '85%'})], style={'display': 'flex'}),
    html.Div([html.Div([], id='graph17', style={'width': '85%'}), html.Div([], id='graph', style={'width': '85%'})], style={'display': 'flex'}),
])

# Callback function definition
@app.callback([Output(component_id='graph1', component_property='children'),
               Output(component_id='graph2', component_property='children'),
               Output(component_id='graph3', component_property='children'),
               Output(component_id='graph4', component_property='children'),
               Output(component_id='graph5', component_property='children'),
               Output(component_id='graph6', component_property='children'),
               Output(component_id='graph7', component_property='children'),
               Output(component_id='graph8', component_property='children'),
               Output(component_id='graph9', component_property='children'),
               Output(component_id='graph10', component_property='children'),
               Output(component_id='graph11', component_property='children'),
               Output(component_id='graph12', component_property='children'),
               Output(component_id='graph13', component_property='children'),
               Output(component_id='graph14', component_property='children'),
               Output(component_id='graph15', component_property='children'),
               Output(component_id='graph16', component_property='children'),
               Output(component_id='graph17', component_property='children'),
               Output(component_id='graph', component_property='children'),
               ],
              [Input(component_id='input-year', component_property='value')],
              )

# Add computation to callback function and return graph
def get_graph(year):
    # Extract the dataframe to plot graph
    df.order_purchase_timestamp = pd.to_datetime(df.order_purchase_timestamp)
    df_year = df[df.order_purchase_timestamp.dt.year == year]

    fig1 = plot1(df_year)
    fig2 = plot2(df_year)
    fig3 = plot3(df_year)
    fig4 = plot4(df_year)
    fig5 = plot5(df_year)
    fig6 = plot6(df_year)
    fig7 = plot7(df_year)
    fig8 = plot8(df_year)
    fig9 = plot9(df_year)
    fig10 = plot10(df_year)
    fig11 = plot11(df_year)
    fig12 = plot12(df_year)
    fig13 = plot13(df_year)
    fig14 = plot14(df_year)
    fig15 = plot15(df_year)
    fig16 = plot16(df_year)
    fig17 = plot17(df_year)
    fig = plotg(df_year)

    # Return dcc.Graph component to the empty division
    return [dcc.Graph(figure=fig1), dcc.Graph(figure=fig2), dcc.Graph(figure=fig3),
            dcc.Graph(figure=fig4), dcc.Graph(figure=fig5), dcc.Graph(figure=fig6),
            dcc.Graph(figure=fig7), dcc.Graph(figure=fig8), dcc.Graph(figure=fig9),
            dcc.Graph(figure=fig10), dcc.Graph(figure=fig11), dcc.Graph(figure=fig12),
            dcc.Graph(figure=fig13), dcc.Graph(figure=fig14), dcc.Graph(figure=fig15),
            dcc.Graph(figure=fig16), dcc.Graph(figure=fig17), dcc.Graph(figure=fig),]


# Plot all the graphs
def plot1(df):
    # Annual Orders
    annual_orders = len(df.order_id.unique())
    # Create indicator
    fig = go.Figure(go.Indicator(
        mode="number",
        value=annual_orders,
        title={'text': "Annual Order Number: "},
    ), layout={'height': 300})
    # Update layout with preferred color and font style
    fig.update_layout(
        font=dict(family='Arial', color='black', size=44),
        paper_bgcolor='#f5f5f5',
        plot_bgcolor='#f5f5f5',
    )
    return fig

def plot2(df):
    # Annual Sales
    annual_sales = df.payment_value.sum()
    # Create indicator
    fig = go.Figure(go.Indicator(
        mode="number",
        value=annual_sales,
        number={'prefix': '$', 'valueformat': ',.0f'},
        title={'text': "Annual Sales"},
    ), layout={'height': 300})
    # Update layout with preferred color and font style
    fig.update_layout(
        font=dict(family='Arial', color='black', size=24),
        paper_bgcolor='#f5f5f5',
        plot_bgcolor='#f5f5f5',
    )
    return fig

def plot3(df):
    # Annual Customers
    annual_customers = len(df.customer_unique_id.unique())
    # Create indicator
    fig = go.Figure(go.Indicator(
        mode="number",
        value=annual_customers,
        title={'text': "Annual Customer Numbers: "},
    ), layout={'height': 300})
    # Update layout with preferred color and font style
    fig.update_layout(
        font=dict(family='Arial', color='black', size=24),
        paper_bgcolor='#f5f5f5',
        plot_bgcolor='#f5f5f5',
    )
    return fig

def plot4(df):
    # Payment type Proportion
    fig = px.pie(df, names='payment_type', title='Payment Types')
    return fig

def plot5(df):
    # Review Score Proportion
    fig = px.pie(df, names='review_score', title='Review Scores')
    return fig

def plot6(df):
    # Stacked Bar Plot of each category of Payment Installments and Payment Value
    payment_data = df[['payment_value', 'payment_installments']].copy()

    # Categorize payment installments
    def categorize_installments(x):
        if x == 1:
            return '1'
        elif x >= 2 and x <= 5:
            return '2-5'
        elif x >= 6 and x <= 10:
            return '6-10'
        else:
            return '>10'

    # Categorize payment value
    def categorize_payment(x):
        if x <= 50:
            return '0-50'
        elif x > 50 and x <= 100:
            return '50-100'
        elif x > 100 and x <= 500:
            return '100-500'
        else:
            return '>500'

    payment_data['installments_label'] = payment_data['payment_installments'].apply(lambda x: categorize_installments(x))
    payment_data['payment_label'] = payment_data['payment_value'].apply(lambda x: categorize_payment(x))
    # Set the order of payment_label categories using CategoricalDtype
    cat_dtype = pd.api.types.CategoricalDtype(categories=["0-50", "50-100", "100-500", ">500"], ordered=True)
    payment_data['payment_label'] = payment_data['payment_label'].astype(cat_dtype)
    # Group data by payment and installment labels
    payment_installments = payment_data.groupby(['payment_label', 'installments_label']).count()['payment_value'].reset_index()
    # Create stacked bar chart
    fig = px.bar(payment_installments, x='payment_label', y='payment_value', color='installments_label',
                         title='Payment Installments Distribution by Payment Value Category',
                         labels={'payment_label': 'Payment Value Category', 'payment_value': 'Count', 'installments_label': 'Payment Installments'}
    )
    return fig

def plot7(df):
    # Top 10 Categories
    top_categories = df.product_category_name_english.value_counts()[:10].reset_index(name="Count")
    fig = px.bar(top_categories, x='Count', y='product_category_name_english', orientation='h', title='Top 10 Product Categories',
                 labels={'order_id': 'Number of Orders', 'product_category_name_english': 'Product Category'},
                 color='Count', color_continuous_scale=['#00FFFF', '#00BFFF', '#1E90FF', '#4169E1', '#0000CD'])
    # Flip the chart to display the largest bars at top
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    return fig

def plot8(df):
    # Top 10 Products
    top_products = df.product_id.value_counts()[:10].reset_index(name="Count")
    top_products.product_id = top_products.product_id.apply(lambda x: 'P' + str(x))
    fig = px.bar(top_products, x='Count', y='product_id', orientation='h', title='Top 10 Products',
                 labels={'Count': 'Number of Orders', 'product_id': 'Product ID'},
                 color='Count', color_continuous_scale=['#00FFFF', '#00BFFF', '#1E90FF', '#4169E1', '#0000CD'])
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    return fig

def plot9(df):
    # Top 10 Sellers
    top_sellers = df.seller_id.value_counts()[:10].reset_index(name="Count")
    top_sellers.seller_id = top_sellers.seller_id.apply(lambda x: 'S' + str(x))
    fig = px.bar(top_sellers, x='Count', y='seller_id', orientation='h', title='Top 10 Sellers',
                 labels={'Count': 'Number of Orders', 'seller_id': 'Seller ID'},
                 color='Count', color_continuous_scale=['#00FFFF', '#00BFFF', '#1E90FF', '#4169E1', '#0000CD'])
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    return fig

def plot10(df):
    # Get customer state counts
    state_counts = df.groupby('customer_state').count()['customer_city']
    # Sort states by number of customers
    state_counts_sorted = state_counts.sort_values(ascending=False).reset_index(name='Count')
    # Create bar chart of customer states
    fig = px.bar(x=state_counts_sorted['customer_state'], y=state_counts_sorted['Count'],
                 title='Distribution of Customers by State', labels={'x': 'State', 'y': 'Number of Customers'})
    fig.update_layout(xaxis={'tickangle': 0})
    return fig

def plot11(df):
    # Get customer city counts
    city_counts = df.groupby(['customer_state', 'customer_city']).size().reset_index(name='count')
    # Get top 19 cities by customer count and group the rest into 'Other' city
    top_cities = city_counts.groupby('customer_city').sum().sort_values(by='count', ascending=False).head(20).reset_index()
    top_cities_list = top_cities['customer_city'].tolist()
    city_counts.loc[~city_counts['customer_city'].isin(top_cities_list), 'customer_city'] = 'Other'
    # Groupby customer state and city again to include 'Other' city
    city_counts = city_counts.groupby(['customer_state', 'customer_city']).sum().reset_index()
    # Sort cities by number of customers
    city_counts_sorted = city_counts.sort_values(by='count', ascending=False)
    city_counts_sorted['customer_city_name'] = range(len(city_counts_sorted))
    # Create bar chart of customer cities
    fig = px.bar(x=city_counts_sorted['customer_city_name'], y=city_counts_sorted['count'],
                 hover_name=city_counts_sorted['customer_city'],
                 title='Distribution of Customers by City', labels={'x': 'City', 'y': 'Number of Customers'})
    fig.update_layout(xaxis={'tickangle': 0})
    return fig

def plot12(df):
    # Customer RFM Model
    # Get relevant columns from dataframe
    rfm_data = df[['customer_unique_id', 'order_purchase_timestamp', 'payment_value', 'review_score']].copy()
    # Convert order_purchase_timestamp to datetime object and calculate recency
    rfm_data['order_purchase_timestamp'] = pd.to_datetime(rfm_data['order_purchase_timestamp'])
    last_order_date = rfm_data['order_purchase_timestamp'].max()
    rfm_data['recency'] = (last_order_date - rfm_data['order_purchase_timestamp']).dt.days
    # Calculate monetary and frequency value
    rfm_data['monetary'] = rfm_data['payment_value']
    rfm_data['frequency'] = rfm_data.groupby('customer_unique_id')['customer_unique_id'].transform('count')
    # Calculate RFM scores and group customers into RFM segments
    rfm_metrics = rfm_data.groupby('customer_unique_id').agg({'recency': 'min', 'frequency': 'max', 'monetary': 'sum'})
    rfm_metrics.rename(columns={'monetary': 'value'}, inplace=True)
    # Define the quantiles
    quantiles = rfm_metrics.quantile(q=[0.25, 0.5, 0.75])
    quantiles = quantiles.to_dict()

    def r_score(x, p, d):
        if x <= d[p][0.25]:
            return 4
        elif x <= d[p][0.5]:
            return 3
        elif x <= d[p][0.75]:
            return 2
        else:
            return 1

    def fm_score(x, p, d):
        if x <= d[p][0.25]:
            return 1
        elif x <= d[p][0.5]:
            return 2
        elif x <= d[p][0.75]:
            return 3
        else:
            return 4

    rfm_metrics['r_quartile'] = rfm_metrics['recency'].apply(r_score, args=('recency', quantiles,))
    rfm_metrics['f_quartile'] = rfm_metrics['frequency'].apply(fm_score, args=('frequency', quantiles,))
    rfm_metrics['m_quartile'] = rfm_metrics['value'].apply(fm_score, args=('value', quantiles,))
    rfm_metrics['RFM_Score'] = rfm_metrics['r_quartile'].map(str) + rfm_metrics['f_quartile'].map(str) + rfm_metrics['m_quartile'].map(str)

    def get_rfm_segment(x):
        if x['RFM_Score'] >= '333':
            return 'Potential Loyalist'
        elif x['RFM_Score'] >= '232':
            return 'Churned Best Customer'
        elif x['RFM_Score'] >= '223':
            return 'Loyal Customer'
        elif x['RFM_Score'] >= '332':
            return 'New Best Customer'
        elif x['RFM_Score'][1:] == '11':
            return 'Promising'
        elif x['RFM_Score'][1:] == '12':
            return 'Needs Attention'
        elif x['RFM_Score'][1:] == '13':
            return 'About to Sleep'
        elif x['RFM_Score'][1:] == '21':
            return 'At Risk'
        elif x['RFM_Score'][1:] == '22':
            return 'Can\'t Lose Them'
        elif x['RFM_Score'][1:] == '23':
            return 'Hibernating'
        else:
            return 'Lost'

    rfm_metrics['RFM_Segment'] = rfm_metrics.apply(get_rfm_segment, axis=1)
    # Plot RFM segments
    rfm_segments = rfm_metrics.groupby('RFM_Segment')['RFM_Score'].count().reset_index().sort_values(by='RFM_Score',ascending=False)
    fig = px.pie(rfm_segments, values='RFM_Score', names='RFM_Segment', title='RFM Segments',
                 labels={'RFM_Score': 'Number of Customers', 'RFM_Segment': 'RFM Segment'})
    return fig

def plot13(df):
    # Percentage of Orders with Different Numbers of Items
    order_items = df[['order_id', 'order_item_id']].copy()
    # Count the number of items in each order by grouping by order_id and counting the order_item_id
    item_counts = order_items.groupby('order_id').count()['order_item_id']
    # Categorize the item counts as 1, 2, 3-5, or >5
    def categorize_items(x):
        if x == 1:
            return '1 item'
        elif x == 2:
            return '2 items'
        elif x >= 3 and x <= 5:
            return '3-5 items'
        else:
            return '>5 items'

    item_counts_cat = item_counts.apply(lambda x: categorize_items(x))
    # Calculate the percentage of each category of item counts
    item_counts_cat_percent = item_counts_cat.value_counts(normalize=True) * 100
    # Create the pie chart
    fig = px.pie(item_counts_cat_percent, values=item_counts_cat_percent.values, names=item_counts_cat_percent.index,
                 title='Number of items in an order',
                 labels={'value': 'Percentage', 'name': 'Item Count'}, hole=0.7)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot14(df):
    # Average Freight Value for each Product Category
    product_data = df[['product_category_name_english', 'freight_value']].copy()
    # Calculate the average freight value for each product category
    category_avg_freight = product_data.groupby('product_category_name_english').mean().reset_index()
    category_avg_freight['product_category'] = range(len(category_avg_freight))
    # Create bar chart
    fig = px.bar(category_avg_freight, x='product_category', y='freight_value',
                 title='Average Freight Value per Product Category', hover_name='product_category_name_english',
                 labels={'product_category_name_english': 'Product Category', 'freight_value': 'Average Freight Value'},
                 color='freight_value')
    # Add a range slider to the x-axis
    fig.update_layout(xaxis={'tickangle': 0})
    return fig

def plot15(df):
    # Monthly PV and UV
    sales_data = df[['order_purchase_timestamp', 'customer_id', 'customer_unique_id', 'payment_value']].copy()
    # Convert timestamp column to month
    sales_data['month'] = pd.to_datetime(sales_data['order_purchase_timestamp']).dt.to_period('M')
    # Group data by month and calculate unique customer counts
    customer_by_month = sales_data.groupby('month')[['customer_id', 'customer_unique_id']].nunique().reset_index()
    customer_by_month.month = customer_by_month.month.astype('str')
    # Create line chart of customer counts by month
    fig = px.line(customer_by_month, x='month', y=['customer_id', 'customer_unique_id'], title='Customer Counts by Month')
    # Update trace labels to PV and UV
    fig.update_traces(name='PV', selector=dict(name='customer_id'))
    fig.update_traces(name='UV', selector=dict(name='customer_unique_id'))
    return fig

def plot16(df):
    # Sales trend of each month
    sales_data = df[['order_purchase_timestamp', 'customer_id', 'customer_unique_id', 'payment_value']].copy()
    # Convert timestamp column to month
    sales_data['month'] = pd.to_datetime(sales_data['order_purchase_timestamp']).dt.to_period('M')
    sales_by_month = sales_data.groupby('month')['payment_value'].sum().reset_index()
    sales_by_month.month = sales_by_month.month.astype('str')
    # Create line chart of total sales by month
    fig = px.line(sales_by_month, x='month', y='payment_value', title='Total Sales by Month')
    return fig

def plot17(df):
    # Order Frequency of each hour in each day-of week
    sales_data = df[['order_purchase_timestamp', 'customer_id', 'customer_unique_id', 'payment_value']].copy()
    # Convert timestamp column to month
    sales_data['month'] = pd.to_datetime(sales_data['order_purchase_timestamp']).dt.to_period('M')
    sales_data['day_of_week'] = pd.to_datetime(sales_data['order_purchase_timestamp']).dt.dayofweek
    sales_data['hour_in_day'] = pd.to_datetime(sales_data['order_purchase_timestamp']).dt.hour
    # Group data by day-of-week and hour-in-day and calculate number of orders
    orders_by_time = sales_data.groupby(['day_of_week', 'hour_in_day'])['order_purchase_timestamp'].count().reset_index()
    # Create heatmap of orders by day-of-week and hour-in-day
    fig = px.imshow(orders_by_time.pivot(index='day_of_week', columns='hour_in_day', values='order_purchase_timestamp'),
                    labels=dict(x="Hour in Day", y="Day of Week"), title='Order Frequency')
    return fig

def plotg(df):
    # Relationship between Freight Value and Shippping Time Period
    shipping_data = df[['freight_value', 'order_purchase_timestamp', 'order_delivered_customer_date']].copy()
    # Calculate the shipping time period for each order
    shipping_data['order_purchase_timestamp'] = pd.to_datetime(shipping_data['order_purchase_timestamp'])
    shipping_data['order_delivered_customer_date'] = pd.to_datetime(shipping_data['order_delivered_customer_date'])
    shipping_data['shipping_time_period'] = (shipping_data['order_delivered_customer_date'] - shipping_data['order_purchase_timestamp']).dt.days
    # Create scatter plot
    fig = px.scatter(shipping_data, x='freight_value', y='shipping_time_period', color='freight_value',
                     title='Relationship between Freight Value and Shipping Time Period',
                     labels={'freight_value': 'Freight Value', 'shipping_time_period': 'Shipping Time Period (Days)'})
    return fig


# Run the app
if __name__ == '__main__':
    app.run_server()