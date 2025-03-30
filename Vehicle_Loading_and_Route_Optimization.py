import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import great_circle, geodesic
from sklearn.cluster import DBSCAN, KMeans
import requests
from datetime import datetime as dt



date = '03.01.24' #'03.01.24' #'01.01.24' 
customer_master_csv = f"C:/Users/ME/OneDrive/Van-Route Optimization/{date}/Customer Master List {date}.csv"
warehouse_csv = f"C:/Users/ME/OneDrive/Van-Route Optimization/{date}/Warehouse Details {date}.csv"
all_customers = pd.read_csv(customer_master_csv)


def calculate_pathway_distance(row, another_parameter1 = None, another_parameter2 = None):
    if another_parameter1 == None:
        #Specify coordinates for two locations
        start_coords = (row['warehouse_latitude'], row['warehouse_longitude'])
        end_coords = (row['customer_latitude'], row['customer_longitude'])

        #Format coordinates for OSRM API
        url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?overview=false"

        #Send the request to the OSRM API
        response = requests.get(url)
        data = response.json()

        #Extract distance in meters
        distance_meters = data['routes'][0]['distance']
        distance_km = distance_meters / 1000  # Convert to kilometers
        # print(f"Distance: {distance_km} km")

        return distance_km
    
    else:
        #Specify coordinates for two locations
        start_coords = row
        end_coords = (another_parameter1, another_parameter2)

        # Format coordinates for OSRM API
        url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?overview=false"

        # Send the request to the OSRM API
        response = requests.get(url)
        data = response.json()


        # Extract distance in meters
        distance_meters = data['routes'][0]['distance']
        distance_km = distance_meters / 1000  # Convert to kilometers
        # print(f"Distance: {distance_km} km")

        return distance_km


def customer_order_clubbing_check(order_data, van_data):
    # Clubbing all customer's orderids, so they can be recorded in the OrderId column
    clubbed_orders_cust_and_orderid_df = pd.DataFrame(order_data[["customer_id", "OrderID"]]).sort_values(by = ["customer_id", "OrderID"], ascending = True)
    clubbed_orders_cust_and_orderid_df["OrderID"] = clubbed_orders_cust_and_orderid_df["OrderID"].astype(str)
    resulting_clubbed_orders = clubbed_orders_cust_and_orderid_df.groupby("customer_id")["OrderID"].apply(lambda a : '-'.join(a)).reset_index()
    resulting_clubbed_orders["invoice_no"] = resulting_clubbed_orders['customer_id'].astype(str) + '-' + resulting_clubbed_orders['OrderID'].astype(str)
    resulting_clubbed_orders_dict = dict(zip(resulting_clubbed_orders["customer_id"], resulting_clubbed_orders["OrderID"]))
    resulting_clubbed_invoice_dict = dict(zip(resulting_clubbed_orders["customer_id"], resulting_clubbed_orders["invoice_no"]))


    max_van_capacity = van_data["Capacity (Volume cm3)"].max()

    grouped_df = order_data.groupby('customer_id').agg({
    'total_volume': 'sum',
    'total_quantity': 'sum',
    'total_weight': 'sum',
    'OrderID': 'nunique',  # Count the number of orders per customer
    'invoice_no' : 'first'
    }).reset_index()

    #Rename the 'OrderID' column to 'order_count'
    grouped_df.rename(columns={'OrderID': 'order_count'}, inplace=True)

    #Split into large and small orders
    large_orders = grouped_df[grouped_df['total_volume'] > max_van_capacity]
    small_orders = grouped_df[grouped_df['total_volume'] <= max_van_capacity]

    #Prepare DataFrames for small orders
    small_orders_df = order_data[order_data['customer_id'].isin(small_orders['customer_id'])].copy()
    small_orders_df['customer_order_type'] = 'within van limit'


    #Group by customer_id for small orders to avoid duplicates
    small_orders_grouped = small_orders_df.groupby('customer_id').agg(
        total_volume=('total_volume', 'sum'),
        total_quantity=('total_quantity', 'sum'),
        total_weight=('total_weight', 'sum'),
        customer_name=('customer_name', 'first'),
        OrderDate=('OrderDate', 'first'),
        Status=('Status', 'first'),
        customer_city=('customer_city', 'first'),
        customer_latitude=('customer_latitude', 'first'),
        customer_longitude=('customer_longitude', 'first'),
        zone=('zone', 'first'),
        Sub_Zone=('Sub_Zone', 'first'),
        warehouse_location=('warehouse_location', 'first'),
        warehouse_latitude=('warehouse_latitude', 'first'),
        warehouse_longitude=('warehouse_longitude', 'first'),
        invoice_no=('invoice_no', 'first'),
        OrderID = ('OrderID', 'first'),
        product_count = ('product_count', 'first'),
        customer_type = ('customer_type', 'first'),
        distance_from_warehouse= ('distance_from_warehouse', 'first'),
        customer_order_type = ('customer_order_type', 'first'),
        merged_products_and_quantity = ('merged_products_and_quantity', ','.join)
    ).reset_index()


    #Merge order_count back into the small orders grouped DataFrame
    small_orders_grouped = small_orders_grouped.merge(grouped_df[['customer_id', 'order_count']], on='customer_id', how='left')


    #Add the has_multiple_orders column
    small_orders_grouped['has_multiple_orders'] = small_orders_grouped['order_count'] > 1

    small_orders_grouped["OrderID"] = small_orders_grouped["customer_id"].map(resulting_clubbed_orders_dict)
    small_orders_grouped["invoice_no"] = small_orders_grouped["customer_id"].map(resulting_clubbed_invoice_dict)

    # Step 8: Prepare DataFrames for large orders and exclude small orders
    large_orders_df = order_data[order_data['customer_id'].isin(large_orders['customer_id'])].copy()
    large_orders_df = large_orders_df[~large_orders_df['customer_id'].isin(small_orders_grouped['customer_id'])]  # Exclude customers in small orders
    large_orders_df['customer_order_type'] = 'exceeding van limit'

    #Combine both DataFrames
    result_df = pd.concat([large_orders_df, small_orders_grouped], ignore_index=True)


    #To check if there are single orders that are beyond the maximum van capacity, so it can be splitted.
    def big_order_splitting(orderdata_df):
        def split_row(row):
            """Splits a single row into two based on total_volume and associated columns."""
            
            # Custom splitting logic based on rounding
            def split_number(number):
                if number % 2 == 0:
                    # Even number: split evenly
                    return number // 2, number // 2
                else:
                    # Odd number: split with rounding
                    half1 = number // 2
                    half2 = half1 + 1
                    return half1, half2

            # Split volume, quantity, and weight using custom split
            volume1, volume2 = split_number(row["total_volume"])
            quantity1, quantity2 = split_number(row["total_quantity"])
            weight1, weight2 = split_number(row["total_weight"])

            # Split merged products and quantities
            merged_items = row["merged_products_and_quantity"].split(',')
            items1, items2 = [], []
            for item in merged_items:
                if ':' not in item:
                    print(f"Skipping invalid item: {item}")
                    continue  # Skip invalid formats
                product, qty = item.split(':')
                try:
                    qty = int(float(qty))  # Ensure quantity is numeric
                    qty1, qty2 = split_number(qty)
                    items1.append(f"{product}:{qty1}")
                    items2.append(f"{product}:{qty2}")
                except ValueError:
                    print(f"Invalid quantity for product {product}: {qty}")
                    continue  # Skip non-numeric quantities

            # Create two new rows
            row1, row2 = row.copy(), row.copy()
            row1.update({
                "total_volume": volume1,
                "total_quantity": quantity1,
                "total_weight": weight1,
                "merged_products_and_quantity": ','.join(items1),
                "invoice_no": f"{row['invoice_no']}_split1",
                "OrderID": f"{row['OrderID']}_split1"
            })
            row2.update({
                "total_volume": volume2,
                "total_quantity": quantity2,
                "total_weight": weight2,
                "merged_products_and_quantity": ','.join(items2),
                "invoice_no": f"{row['invoice_no']}_split2",
                "OrderID": f"{row['OrderID']}_split2"
            })
            return row1, row2

        # Process rows that exceed max van capacity
        while (orderdata_df["total_volume"] > max_van_capacity).any():
            # Separate large orders and those within capacity
            large_orders = orderdata_df[orderdata_df["total_volume"] > max_van_capacity]
            small_orders = orderdata_df[orderdata_df["total_volume"] <= max_van_capacity]
            
            split_entries = []

            for _, row in large_orders.iterrows():
                row1, row2 = split_row(row)
                split_entries.extend([row1, row2])

            # Combine small orders and split entries
            split_df = pd.DataFrame(split_entries)
            orderdata_df = pd.concat([small_orders, split_df], ignore_index=True)


        # Return the DataFrame, which will be split properly
        return orderdata_df

    result_df = big_order_splitting(result_df)

    return result_df


def stock_check(stock_csv, melted_orders_df):

    melted_orders_df = pd.DataFrame(melted_orders_df[(melted_orders_df["Status"].str.title() == "Approved") & (melted_orders_df['product_code'].notna())])

    stock_df = pd.read_csv(stock_csv)
    # stock_df.dropna(inplace = True)

    stock_df["Unrestricted Stock"] = stock_df["Unrestricted Stock"].apply(lambda a : int(float(a.strip("CAR").strip(" "))))

    stock_df.drop_duplicates(keep = "first", ignore_index = True, inplace = True)

    if 'Volume of ctn' not in stock_df.columns:
        stock_df['Volume of ctn'] = np.nan
    else:
        pass

    grouped_stock_df = pd.DataFrame(stock_df.groupby(["Material", "Material Description"])[["Unrestricted Stock", "Volume of ctn"]].sum()).reset_index()
    grouped_stock_df["Material"] = grouped_stock_df["Material"].astype(object)

    grouped_stock_df.rename(columns = {
        "Unrestricted Stock" : "Available_Quantity",
        "Volume of ctn" : "Total_Available_Volume"
        }, inplace = True)

    #extracting priority customers, so we can prioritize the stock check accordingly
    all_customers["Contact Code"] = all_customers["Contact Code"].astype(int)
    priority_customers_list = list(all_customers[all_customers["IsPriorityCustomer"] == "Yes"]["Contact Code"])

    melted_orders_df["Distributor Code"] = melted_orders_df["Distributor Code"].astype(int)

    #Giving preference to priority customers. Within that, we also sort by earliest dates
    top_priority_melted_orders_df = melted_orders_df[melted_orders_df["Distributor Code"].isin(priority_customers_list)].sort_values(by = "SO Date", ascending = True)

    bottom_priority_melted_orders_df = melted_orders_df[~melted_orders_df["Distributor Code"].isin(priority_customers_list)].sort_values(by = "SO Date", ascending = True)

    melted_orders_df = pd.concat([top_priority_melted_orders_df, bottom_priority_melted_orders_df], ignore_index = True)


    try:
        melted_orders_df["quantity"] = melted_orders_df["quantity"].str.replace(',', '').astype(float)
    except:
        pass
    melted_orders_df["product_code"] = melted_orders_df["product_code"].astype(str)
    grouped_stock_df["Material"] = grouped_stock_df["Material"].astype(str)

    


    #computing the stock checks, iterating through and keeping track of the qty
    fulfillment_tracker = []
    fulfilled_quantity = []

    for idx, row in melted_orders_df.iterrows():
        selected_product = row[1]
        selected_product_quantity = row[12]

        #Filter the grouped stock for the selected product
        selected_grouped_stock = grouped_stock_df[grouped_stock_df["Material"] == selected_product]
        filtered_selected_grouped_stock = selected_grouped_stock[selected_grouped_stock["Available_Quantity"] > 0]
       

        if not filtered_selected_grouped_stock.empty:
            current_stock_quantity = filtered_selected_grouped_stock["Available_Quantity"].sum()  # Sum to get total available

            if selected_product_quantity > current_stock_quantity:
                fulfillment_tracker.append("Partial Fulfillment")
                fulfilled_quantity.append(current_stock_quantity)

                #Deduct total available quantity
                grouped_stock_df.loc[grouped_stock_df["Material"] == selected_product, "Available_Quantity"] -= current_stock_quantity
            else:
                fulfillment_tracker.append("Complete Fulfillment")
                fulfilled_quantity.append(selected_product_quantity)
                grouped_stock_df.loc[grouped_stock_df["Material"] == selected_product, "Available_Quantity"] -= selected_product_quantity
        else:
            fulfillment_tracker.append("No Fulfillment")
            fulfilled_quantity.append(0)


    melted_orders_df["Fulfillment_Status"] = fulfillment_tracker
    melted_orders_df["Fulfillment_Quantity"] = fulfilled_quantity
    melted_orders_df["Initial_Quantity"] = melted_orders_df["quantity"]

 
    #saving orders to a file, that either had partial fulfillment or no fulfillment
    prep_failed_stock_check_df = pd.DataFrame(melted_orders_df[melted_orders_df["Fulfillment_Status"] != "Complete Fulfillment"])
    prep_failed_stock_check_df["Distributor Code"] = prep_failed_stock_check_df["Distributor Code"].astype(int)

    all_customers_wanted_columns = pd.DataFrame(all_customers[["Contact Code", "IsPriorityCustomer"]])

    failed_stock_check_df = pd.merge(left = prep_failed_stock_check_df, right = all_customers_wanted_columns, left_on = "Distributor Code", right_on = "Contact Code", how = "inner", suffixes = ('', '_new'))

    failed_stock_check_df.drop(labels = ["Contact Code"], axis = 1, inplace = True)
    failed_stock_check_df.to_csv(failed_stock_csv, index = False)

    #overriding the initially placed quantity, with the output after the stock check was computed
    melted_orders_df["quantity"] = melted_orders_df["Fulfillment_Quantity"]

    
    #keeping a version of the unfiltered melted orders df
    unfiltered_melted_orders_df = melted_orders_df.copy()

    unfiltered_melted_orders_df.to_csv(f"C:/Users/ME/OneDrive/Van-Route Optimization/{date}/Unfiltered Melted Orders {date}.csv", index = False)

    #deciding to proceed with orders that had either complete or partial fulfillment
    melted_orders_df = melted_orders_df[melted_orders_df["Fulfillment_Status"] != "No Fulfillment"]

    return melted_orders_df


def load_and_process_orders_csv(orders_csv):
    temp_orders_df = pd.read_csv(orders_csv, header=None)
    # temp_orders_df.dropna(axis = 0, inplace = True)
    van_df = pd.read_csv(vans_csv)
    temp_orders_df = pd.DataFrame(temp_orders_df.iloc[:, :-1])
    first_header_row = temp_orders_df.iloc[0]
    second_header_row = temp_orders_df.iloc[1]
    normal_headers = temp_orders_df.iloc[2]

    combined_header = []
    for idx, col in enumerate(normal_headers):
        if pd.isna(first_header_row[idx]) or first_header_row[idx].lower().strip() == 'item code': 
            combined_header.append(col.strip())
        else:  
            combined_header.append(first_header_row[idx].strip())


    orders_df = pd.DataFrame(temp_orders_df.iloc[3:, :])
    orders_df.columns = combined_header
    

    first_product_code_index = next(i for i, col in enumerate(orders_df.columns) if all(c.isdigit() for c in col))
    order_columns = orders_df.columns[ :first_product_code_index] 
    product_columns = orders_df.columns[first_product_code_index: ] 
    melted_orders_df = orders_df.melt(id_vars=order_columns, value_vars=product_columns, var_name='product_code', value_name='quantity')
    melted_orders_df = pd.DataFrame(melted_orders_df[melted_orders_df['quantity'].notna() & (melted_orders_df['quantity'] != 0)].reset_index(drop=True))

    melted_orders_df = pd.DataFrame(melted_orders_df[(melted_orders_df["Status"].str.title() == "Approved") & (melted_orders_df['product_code'].notna())])
    melted_orders_df["quantity"] = melted_orders_df["quantity"].str.replace(',', '').astype(float)

    try:
        melted_orders_df['SO Date'] = melted_orders_df['SO Date'].apply(lambda a : dt.strptime(a, '%m/%d/%Y'))
        melted_orders_df['SO APPR Date'] = melted_orders_df['SO APPR Date'].apply(lambda a : dt.strptime(a, '%m/%d/%Y'))
    except:
        melted_orders_df['SO Date'] = melted_orders_df['SO Date'].apply(lambda a : dt.strptime(a, '%d/%m/%Y'))
        melted_orders_df['SO APPR Date'] = melted_orders_df['SO APPR Date'].apply(lambda a : dt.strptime(a, '%d/%m/%Y'))


    melted_orders_df = melted_orders_df.groupby(["Distributor Code", "product_code"]).agg(
        Zone = ('Zone', 'first'),
        SubZone = ('Sub Zone', 'first'),
        CustomerName = ('Customer Name', 'first'),
        SOAPPRDate = ('SO APPR Date', 'first'),
        SODate = ('SO Date', 'first'),
        SONumber = ('SO Number', ','.join),
        CustomerCity = ('Customer City', 'first'),
        Status = ('Status', 'first'),
        Latitude = ('Latitude', 'first'),
        Longitude = ('Longitude', 'first'),
        quantity = ('quantity', 'sum')
    ).reset_index()

    melted_orders_df.rename(columns = {
        'SubZone' : 'Sub Zone',
        'CustomerName' : 'Customer Name',
        'SOAPPRDate' : 'SO APPR Date',
        'SODate' : 'SO Date',
        'SONumber' : 'SO Number',
        'CustomerCity' : 'Customer City'
    }, inplace = True)


    #added stock check here as well
    melted_orders_df = stock_check(stock_csv, melted_orders_df)

    return orders_df , melted_orders_df


def transform_data(melted_orders_df, products_csv, warehouse_csv, vans_csv, customer_master_csv, flag = 2):
    products_df = pd.read_csv(products_csv)
    warehouse_df = pd.read_csv(warehouse_csv)
    # warehouse_df.dropna(inplace = True)
    van_df = pd.read_csv(vans_csv)
    customer_master_df = pd.read_csv(customer_master_csv)
    customer_master_df['Contact Code'] = customer_master_df['Contact Code'].astype(int)
    
    # Load van data and find max capacity
    van_df['total_loaded_volume'] = 0
    van_df = van_df[van_df['VehicleNumber'].notna() & (van_df['VehicleNumber'] != '')]
    van_df = van_df.sort_values(by='Capacity (Volume cm3)', ascending=False)

    melted_orders_df['product_code'] = melted_orders_df['product_code'].astype(str)
    melted_orders_df['Distributor Code'] = melted_orders_df['Distributor Code'].astype(int)

    
    # Handle quantity conversion
    try:
        melted_orders_df['quantity'] = melted_orders_df['quantity'].str.replace(',', '').str.strip().astype(float)
    except:
        pass

    # Prepare product data
    products_df['Material Code'] = products_df['Material Code'].astype(str)
    products_df['Volume (cm3)'] = products_df['Volume (cm3)'].astype(str).str.replace(',', '').str.strip()
    
    # Merge orders with products and customer data
    temp_merged_df = pd.merge(melted_orders_df, products_df, how='left', left_on='product_code', right_on='Material Code')
    merged_df = pd.merge(temp_merged_df, customer_master_df, how='left', left_on='Distributor Code', right_on='Contact Code', suffixes=('', '_new'))

    merged_df['warehouse_location'] = warehouse_df['Location'][0]
    merged_df['warehouse_latitude'] = warehouse_df['Latitude'][0].astype(float)
    merged_df['warehouse_longitude'] = warehouse_df['Longitude'][0].astype(float)
    merged_df['Latitude'] = merged_df['Latitude'].astype(float)
    merged_df['Longitude'] = merged_df['Longitude'].astype(float) 

    # Rename columns for final order DataFrame
    final_order_df = merged_df.rename(columns={
        'SO Number': 'OrderID',
        'Distributor Code': 'customer_id',
        'Customer Name': 'customer_name',
        'customer_type': 'customer_type',
        'Latitude': 'customer_latitude',
        'Longitude': 'customer_longitude',
        'Material Code': 'material_code',
        'Material Description': 'product_description',
        'Volume (cm3)': 'product_volume',
        'Weight in Kgs': 'product_weight',
        'SO Date': 'OrderDate',
        'CustomerType': 'customer_type',
        'Customer City': 'customer_city'
    })


    # Filter for approved status and calculate total volume
    final_order_df = pd.DataFrame(final_order_df[(final_order_df['Status'].str.lower() == 'approved') & (final_order_df['material_code'].notna())])
    final_order_df['total_volume'] = final_order_df['product_volume'].astype(float) * final_order_df['quantity'].astype(float)
    final_order_df['invoice_no'] = final_order_df['customer_id'].astype(str) + '-' + final_order_df['OrderID'].astype(str)
    
    # To know the type of quantity to sum up. (fulfilled or initial depending on the point this function is called)
    if flag == 1:   #flag 1 means we are running this function for order raising purpose
        quantity_reference = "Initial_Quantity"
    else:
        quantity_reference = "quantity"

    final_order_df['merged_products_and_quantity'] = final_order_df['Short'] + ':' + final_order_df['Fulfillment_Quantity'].astype(str)

    # Group by invoice_no and aggregate
    final_order_df = final_order_df.groupby('invoice_no').agg(
        customer_id=('customer_id', 'first'),
        OrderID=('OrderID', 'first'),
        customer_name=('customer_name', 'first'),
        OrderDate=('OrderDate', 'first'),
        Status=('Status', 'first'),
        customer_city=('customer_city', 'first'),
        customer_latitude=('customer_latitude', 'first'),
        customer_longitude=('customer_longitude', 'first'),
        zone=('Zone', 'first'),
        Sub_Zone=('Sub Zone', 'first'),
        warehouse_location=('warehouse_location', 'first'),
        warehouse_latitude=('warehouse_latitude', 'first'),
        warehouse_longitude=('warehouse_longitude', 'first'),
        total_volume=('total_volume', 'sum'),
        total_quantity=(f'{quantity_reference}', 'sum'),
        total_weight=('product_weight', 'sum'),
        product_count = ('product_code', 'nunique'),
        merged_products_and_quantity = ('merged_products_and_quantity', ','.join),
        customer_type=('customer_type', 'first')
    ).reset_index()

        
    final_order_df['distance_from_warehouse'] = final_order_df.apply(calculate_pathway_distance, axis=1)

    # print("\n")
    # print(final_order_df.columns)
    # print(final_order_df.shape)

    # print(final_order_df['customer_latitude'].unique())
    # print(final_order_df['customer_latitude'].nunique())

    # print("Here 88")
    # quit()

    #passing it through the order clubbing check function
    accra_final_order_df = pd.DataFrame(final_order_df[final_order_df['zone'].str.lower() == 'accra'])
    upcountry_final_order_df = pd.DataFrame(final_order_df[final_order_df['zone'].str.lower() != 'accra'])

    final_order_df_merger = pd.DataFrame()

    if not accra_final_order_df.empty:
        accra_van_df = van_df[van_df['Capacity (Volume cm3)'] <= 30000000]
        final_order_df_1 = customer_order_clubbing_check(accra_final_order_df, accra_van_df)
        final_order_df_merger = pd.concat([final_order_df_1, final_order_df_merger], ignore_index = True)


    if not upcountry_final_order_df.empty:
        final_order_df_2 = customer_order_clubbing_check(upcountry_final_order_df, van_df)
        final_order_df_merger = pd.concat([final_order_df_2, final_order_df_merger], ignore_index = True)

    

    return final_order_df_merger, van_df


def optimal_kmeans_clusters(coords, max_k=10): #max_k was formerly 7
    #Compute inertia (sum of squared distances) for a range of k values
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(coords)
        distortions.append(kmeans.inertia_)
    
    #Automatically detect the "elbow" (where inertia starts decreasing slower)
    #Compute the second derivative to find the elbow point
    deltas = np.diff(distortions)
    deltas2 = np.diff(deltas)  #2nd derivative
    #optimal_k = np.argmin(deltas2) + 2  #Add 2 because we applied diff() twice

    #This simply gives the k number that returned the least inertia
    optimal_k = distortions.index(min(distortions)) + 1 #Added 1 because python numbering/indexing starts from zero

    return optimal_k


def cluster_customers(final_order_df, eps_km=5, min_samples=2, max_k=10):   #max_k formerly 7
    kms_per_radian = 6371.0088  
    epsilon = eps_km / kms_per_radian 
    db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    customer_coords = np.radians(final_order_df[['customer_latitude', 'customer_longitude']])

    customer_coords['customer_latitude'] = customer_coords['customer_latitude'].astype(str)
    customer_coords = customer_coords[customer_coords['customer_latitude'].str.lower() != 'nan']

    customer_coords['customer_latitude'] = customer_coords['customer_latitude'].astype(float)

    #Temp Resolution for final_order_df
    final_order_df['customer_latitude'] = final_order_df['customer_latitude'].astype(str)
    final_order_df = final_order_df[final_order_df['customer_latitude'].str.lower() != 'nan']
    final_order_df['customer_latitude'] = final_order_df['customer_latitude'].astype(float)

    final_order_df['cluster'] = db.fit_predict(customer_coords)

    final_order_df['is_noise'] = (final_order_df['cluster'] == -1).astype(int)
    noise_points = pd.DataFrame(final_order_df[final_order_df['is_noise'] == 1])
    clustered_points =pd.DataFrame(final_order_df[final_order_df['is_noise'] == 0])

    #To know the max distance of the noise point so as to give an idea how many clusters to pass as argument for kmeans 
    max_noise_distance = noise_points["distance_from_warehouse"].max()

    #Using 70th percentile to consider where most of the points are within (neglecting 30%)
    max_noise_distance_benchmark = noise_points["distance_from_warehouse"].quantile(0.70)  

    #Using 25km to give a sense of how clubbed we want the clusters to be
    proposed_number_of_clusters = round(max_noise_distance_benchmark / 30)
    if proposed_number_of_clusters <= 1:  #incase the division cannot be rounded up to 1
        proposed_number_of_clusters = 1
 
    if not noise_points.empty and not clustered_points.empty:
        noise_coords = np.radians(noise_points[['customer_latitude', 'customer_longitude']].values)
        clustered_coords = np.radians(clustered_points[['customer_latitude', 'customer_longitude']].values)
        optimal_clusters = optimal_kmeans_clusters(noise_coords, max_k=proposed_number_of_clusters)   #formerly clustered_coords
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        noise_clusters = kmeans.fit_predict(noise_coords)
        final_order_df.loc[noise_points.index, 'cluster'] = noise_clusters + final_order_df['cluster'].max() + 1
    
    return final_order_df


def get_clusters(final_order_df, eps_km=5, min_samples=2, max_k=10):
    supermarket_df = final_order_df[final_order_df['customer_type'].str.lower() == 'supermarket'].copy()
    non_supermarket_df = final_order_df[final_order_df['customer_type'].str.lower() != 'supermarket'].copy()
    non_supermarket_clusters = pd.DataFrame()
    supermarket_clusters = pd.DataFrame()
    
    if not non_supermarket_df.empty:
        non_supermarket_clusters = cluster_customers(non_supermarket_df, eps_km, min_samples, max_k)
        
    if not supermarket_df.empty:    
        supermarket_clusters = cluster_customers(supermarket_df, eps_km, min_samples, max_k)

    if not supermarket_clusters.empty and not non_supermarket_clusters.empty:
        max_non_supermarket_cluster = non_supermarket_clusters['cluster'].max()
        supermarket_clusters['cluster'] += max_non_supermarket_cluster + 1

    final_order_df = pd.concat([non_supermarket_clusters, supermarket_clusters], ignore_index=True)

    return final_order_df


def create_sub_clusters(final_order_df, distance_threshold_km=5):
    conditions = [
        (final_order_df['customer_type'].str.lower() == 'supermarket'),
        (final_order_df['distance_from_warehouse'] <= distance_threshold_km) & (final_order_df['customer_type'].str.lower() != 'supermarket'),
        (final_order_df['distance_from_warehouse'] > distance_threshold_km) & (final_order_df['customer_type'].str.lower() != 'supermarket')
            & (final_order_df['is_noise'] == 0),
        (final_order_df['distance_from_warehouse'] > distance_threshold_km) & (final_order_df['customer_type'].str.lower() != 'supermarket')
            & (final_order_df['is_noise'] != 0)    
    ]

    choices = ['supermarket', 'near', 'far_1', 'far_2']
    final_order_df['sub_cluster'] = np.select(conditions, choices, default='far_2')

    return final_order_df


def calculate_cluster_volume_weight(final_order_df):
    cluster_info_df = final_order_df.groupby('cluster').agg({
        'total_volume': 'sum',
        'total_weight': 'sum',
        'total_quantity': 'sum',
        'customer_latitude': 'mean',
        'customer_longitude': 'mean',
        'warehouse_latitude': 'first',
        'warehouse_longitude': 'first'
    }).reset_index()
    
    cluster_info_df['warehouse_distance'] = cluster_info_df.apply(calculate_pathway_distance, axis=1)
    
    return cluster_info_df
 

def aggregate_metrics(customer_van_df, column_name, type, agg='sum', flag=1, percent=0):
    ops_type = {'volume': 'total_loaded_volume', 'quantity': 'total_loaded_quantity', 'van_max_distance': 'van_max_distance',
                'van_min_distance': 'van_min_distance'}
    
    #For cases where the customer_van_df isn't a dataframe due to vans not being available
    try:
        customer_van_df_colums = customer_van_df.columns
    except:
        customer_van_df = pd.DataFrame({
            'customer_id' : [np.nan],
            'invoice_no' : [np.nan],
            'invoice_volume' : [0],
            'van_id' : [np.nan],
            'van_capacity' : [0],
            'total_loaded_volume' : [0],
            'percent_full' : [0]
        })

        return customer_van_df
    

    if ops_type.get(type) not in customer_van_df_colums:
        customer_van_df[ops_type.get(type)] = 0  
    
    if flag == 1:
        van_usage = customer_van_df.groupby(['van_id', 'ride_batch']).agg({column_name: agg}).reset_index()
        van_usage.rename(columns={column_name: ops_type.get(type)}, inplace=True)
        customer_van_df = customer_van_df.merge(van_usage, on=['van_id', 'ride_batch'], how='left', suffixes=('', '_new'))
        customer_van_df[ops_type.get(type)] = customer_van_df[f"{ops_type.get(type)}_new"].fillna(0)
        customer_van_df.drop(f"{ops_type.get(type)}_new", axis=1, inplace=True)      
    elif flag == 0:
        van_usage = customer_van_df.groupby(['van_id']).agg({column_name: agg}).reset_index()
        van_usage.rename(columns={column_name: ops_type.get(type)}, inplace=True)
        customer_van_df = customer_van_df.merge(van_usage, on=['van_id'], how='left', suffixes=('', '_new'))
        customer_van_df[ops_type.get(type)] = customer_van_df[f"{ops_type.get(type)}_new"].fillna(0)
        customer_van_df.drop(f"{ops_type.get(type)}_new", axis=1, inplace=True)
        
    if percent == 1:
        customer_van_df['percent_full'] = (customer_van_df['total_loaded_volume'] / customer_van_df['van_capacity']) * 100


    return customer_van_df
    
    
def select_vans_for_cluster(final_order_df, available_vans, max_distance_km=150, ride_batch_check=None, cluster_checker1=1, sub_cluster_checker1="None"):
    # Warehouse information setup
    warehouse_location = final_order_df['warehouse_location'].iloc[0]
    final_order_df = final_order_df.sort_values(by=['total_volume'], ascending=False)
    warehouse_info_df = pd.read_csv(warehouse_csv)
    warehouse_info_df = warehouse_info_df[warehouse_info_df["Type Of Fulfillment Center"].str.upper() == "FACTORY WH"]

    warehouse_latitude = warehouse_info_df['Latitude'].iloc[0]
    warehouse_longitude = warehouse_info_df['Longitude'].iloc[0]
    sub_cluster_value = final_order_df['sub_cluster'].iloc[0]
    max_customer = {'near': 5, 'far_1': 5, 'far_2': 3, 'supermarket': 7}.get(sub_cluster_value, 5)
    accra_customer_list = list(final_order_df[final_order_df["zone"].str.title() == "Accra"]["customer_id"])

    if final_order_df['distance_from_warehouse'].max() >= max_distance_km:
        available_vans = pd.DataFrame(available_vans[available_vans['Capacity (Volume cm3)'] > 19000000])
    
    if warehouse_location.title() == 'Accra':
        available_vans = pd.DataFrame(available_vans[available_vans['Capacity (Volume cm3)'] < 30000000])

    # # Initialize customer van mapping
    # customer_van_mapping = pd.DataFrame(columns=['customer_id', 'invoice_no', 'invoice_volume', 'van_id', 'van_capacity'])

    def assign_vans(orders, available_vans, max_customer, ride_batch_check, cluster_checker1=1, sub_cluster_checker1="None"):
        # # Initialize customer van mapping
        customer_van_mapping = pd.DataFrame(columns=['customer_id', 'invoice_no', 'invoice_volume', 'van_id', 'van_capacity', 'customer_latitude', 'customer_longitude', 'customer_city'])


        available_vans.reset_index(drop=True, inplace=True)
        if available_vans.empty:
            for _, order in orders.iterrows():
                if order['invoice_no'] not in customer_van_mapping['invoice_no'].values:
                    customer_van_mapping = customer_van_mapping.append({
                        'customer_id': order['customer_id'],
                        'invoice_no': order['invoice_no'],
                        'invoice_volume': order['total_volume'],
                        'van_id': 'no van available',
                        'van_capacity': 1,
                        'customer_latitude' : order['customer_latitude'],
                        'customer_longitude' : order['customer_longitude'],
                        'customer_city' : order['customer_city']
                    }, ignore_index=True)


            customer_van_mapping['van_id'] = customer_van_mapping['van_id'].fillna('no van available').replace('', 'no van available')
            return customer_van_mapping              

        
        def load_orders_to_van(order_list, available_vans, category_of_customers = None):
            if category_of_customers == "Accra":
                available_vans = available_vans[available_vans["Capacity (Volume cm3)"] < 30000000]

            available_vans.sort_values(by="Capacity (Volume cm3)", ascending=True, inplace = True)
            available_vans.reset_index(inplace=True, drop=True)
            # Initialize customer van mapping
            customer_van_mapping = pd.DataFrame(columns=['customer_id', 'invoice_no', 'invoice_volume', 'van_id', 'van_capacity', 'customer_latitude', 'customer_longitude', 'customer_city'])

            current_van_index = 0
            skipped_orders = order_list.copy()

            while current_van_index < len(available_vans) and not skipped_orders.empty:
                current_van_capacity = available_vans.iloc[current_van_index]['Capacity (Volume cm3)'].astype(float)
                current_van_id = available_vans.iloc[current_van_index]['VehicleNumber']
                current_van_volume_used = available_vans.iloc[current_van_index]['total_loaded_volume']
                loaded_customers = 0
                loaded_customer_ids = set()
                
                remaining_orders = skipped_orders.copy()
                skipped_orders = pd.DataFrame(columns=order_list.columns)

                total_loaded_quantity = 0
                distance_to_customer_list = []
                
                # Load customers into van without interruptions
                for idx, order in remaining_orders.iterrows():
                    customer_id = order['customer_id']
                    invoice_volume = order['total_volume']
                    # Extract customer latitude and longitude
                    customer_latitude = order['customer_latitude']
                    customer_longitude = order['customer_longitude']
                    customer_city = order['customer_city']

                    if (current_van_volume_used + invoice_volume > current_van_capacity) or (loaded_customers >= max_customer):
                        order_df = pd.DataFrame([order])

                        for col in order_df.select_dtypes(include=['object']):
                            if order_df[col].dropna().isin([True, False]).all():
                                order_df[col] = order_df[col].astype(bool)

                        skipped_orders = pd.concat([skipped_orders, order_df], ignore_index=True)

                        # skipped_orders = pd.concat([skipped_orders, pd.DataFrame([order])], ignore_index=True)
                        continue

                    current_van_volume_used += invoice_volume

                    if customer_id not in loaded_customer_ids:
                        loaded_customers += 1
                        loaded_customer_ids.add(customer_id)

                    customer_van_mapping = customer_van_mapping.append({
                        'customer_id': order['customer_id'],
                        'invoice_no': order['invoice_no'],
                        'invoice_volume': order['total_volume'],
                        'van_id': current_van_id,
                        'van_capacity': current_van_capacity,
                        'customer_latitude' : customer_latitude,
                        'customer_longitude' : customer_longitude,
                        'customer_city' : customer_city
                    }, ignore_index=True)



                    distance_to_customer = calculate_pathway_distance((warehouse_latitude, warehouse_longitude), another_parameter1=customer_latitude, another_parameter2 = customer_longitude)

                    distance_to_customer_list.append(distance_to_customer)
                    total_loaded_quantity += order['total_quantity']


                #Check the conditions before finalizing the van assignment
                if total_loaded_quantity <= 100 and max(distance_to_customer_list, default = 0) >= 10:
                    current_van_index += 1

                    customer_van_mapping.loc[customer_van_mapping['van_id'] == current_van_id, 'van_id'] = 'No Van'
                    customer_van_mapping.loc[customer_van_mapping['van_id'] == 'No Van', 'van_capacity'] = 1

                    continue  # Move to the next van
                

                #Check route constraints after loading
                #Check route constraints after loading
                
                removed_customers = check_route_constraints(customer_van_mapping[customer_van_mapping['van_id'] == current_van_id], warehouse_latitude, warehouse_longitude, max_distance_from_4th_dropoff=7, passed_van_id = current_van_id)
                
                if removed_customers:  # If there are any removed customers
                    #Remove these customers from customer_van_mapping
                    customer_van_mapping = customer_van_mapping[~customer_van_mapping['customer_id'].isin(removed_customers)]

                #Update the van's loaded volume
                available_vans.at[current_van_index, 'total_loaded_volume'] = current_van_volume_used
                current_van_index += 1

            return customer_van_mapping


        def check_route_constraints(van_orders, warehouse_latitude, warehouse_longitude, max_distance_from_4th_dropoff=7, passed_van_id = 0):
            route = []
            removed_customers = []
            remaining_customers = van_orders.copy()

            
            #Return empty list if there are no customers to check
            if van_orders.empty:
                return []

            #Separate OKAISHIE and other customers
            okaishie_customers = pd.DataFrame(remaining_customers[remaining_customers['customer_city'].str.upper() == 'OKAISHIE'])
            other_customers = pd.DataFrame(remaining_customers[remaining_customers['customer_city'].str.upper() != 'OKAISHIE'])

            #Calculate distance to warehouse for OKAISHIE customers, find the closest starting point
            if not okaishie_customers.empty:
                okaishie_customers['distance_to_warehouse'] = okaishie_customers.apply(
                    lambda x: calculate_pathway_distance(
                        (warehouse_latitude, warehouse_longitude),
                        another_parameter1=x['customer_latitude'], 
                        another_parameter2=x['customer_longitude']
                    ),
                    axis=1
                )

                #Find the closest OKAISHIE customer to the warehouse and start the route
                first_okaishie_customer = okaishie_customers.loc[okaishie_customers['distance_to_warehouse'].idxmin()]
                first_distance = first_okaishie_customer['distance_to_warehouse']
                route.append([first_okaishie_customer['customer_id'], first_distance])
                okaishie_customers = okaishie_customers.drop(first_okaishie_customer.name)

                #Route OKAISHIE customers iteratively
                while not okaishie_customers.empty:
                    last_customer = route[-1]  # Get the last customer in the route
                    last_customer_location = (
                        van_orders.loc[van_orders['customer_id'] == last_customer[0], 'customer_latitude'].values[0],
                        van_orders.loc[van_orders['customer_id'] == last_customer[0], 'customer_longitude'].values[0]
                    )

                    #Calculate distance to the last customer in OKAISHIE
                    okaishie_customers['distance_to_last_customer'] = okaishie_customers.apply(
                        lambda x: calculate_pathway_distance(
                            last_customer_location, 
                            another_parameter1=x['customer_latitude'], 
                            another_parameter2=x['customer_longitude']
                        ),
                        axis=1
                    )

                    #Find the closest OKAISHIE customer to the last OKAISHIE customer in route
                    next_customer = okaishie_customers.loc[okaishie_customers['distance_to_last_customer'].idxmin()]
                    distance_to_next_customer = next_customer['distance_to_last_customer']

                    #Append the next OKAISHIE customer to the route
                    route.append([next_customer['customer_id'], distance_to_next_customer])

                    #Apply the distance constraint if we're past the third OKAISHIE customer
                    if len(route) > 3 and distance_to_next_customer > max_distance_from_4th_dropoff:
                        # Add all remaining customers (both OKAISHIE and other) to removed list
                        removed_customers.extend(okaishie_customers['customer_id'].tolist())
                        removed_customers.extend(other_customers['customer_id'].tolist())
                        return removed_customers  # Exit early since all remaining customers are removed

                    #Drop the next customer from okaishie_customers only after passing the distance check
                    okaishie_customers = okaishie_customers.drop(next_customer.name)

            
            #Process remaining customers, starting from the last OKAISHIE customer if any, or the warehouse if none
            if route:
                last_customer_location = (
                    van_orders.loc[van_orders['customer_id'] == route[-1][0], 'customer_latitude'].values[0],
                    van_orders.loc[van_orders['customer_id'] == route[-1][0], 'customer_longitude'].values[0]
                )
            else:
                last_customer_location = (warehouse_latitude, warehouse_longitude)

            #Calculate routes for non-OKAISHIE customers
            while not other_customers.empty:
                other_customers['distance_to_last_customer'] = other_customers.apply(
                    lambda x: calculate_pathway_distance(
                        last_customer_location, 
                        another_parameter1=x['customer_latitude'], 
                        another_parameter2=x['customer_longitude']
                    ),
                    axis=1
                )

                #Find the closest non-OKAISHIE customer to the last point in the route
                next_customer = other_customers.loc[other_customers['distance_to_last_customer'].idxmin()]
                distance_to_next_customer = next_customer['distance_to_last_customer']

                
                #Append the next customer to the route and remove from remaining customers
                route.append([next_customer['customer_id'], distance_to_next_customer])

                #Apply the distance constraint from the 4th drop-off onward
                if len(route) > 3 and distance_to_next_customer > max_distance_from_4th_dropoff:
                    # Add remaining customers to removed list if the constraint is exceeded
                    removed_customers.extend(other_customers['customer_id'].tolist())
                    break

                #Drop the next customer from other_customers only after passing the distance check
                other_customers = other_customers.drop(next_customer.name)

                #Update the last_customer_location to the latest customer's location in the route
                last_customer_location = (
                    next_customer['customer_latitude'],
                    next_customer['customer_longitude']
                )

            return removed_customers



        #Process orders based on location groups
        accra_orders = orders[orders['customer_id'].isin(accra_customer_list)]
        upcountry_orders = orders[~orders['customer_id'].isin(accra_customer_list)]

        if not accra_orders.empty:
            customer_van_mapping = load_orders_to_van(accra_orders, available_vans, category_of_customers = "Accra")
            used_vans = customer_van_mapping['van_id'].unique()
            available_vans = available_vans[~available_vans['VehicleNumber'].isin(used_vans)]

        if not upcountry_orders.empty:
            customer_van_mapping = pd.concat([customer_van_mapping, load_orders_to_van(upcountry_orders, available_vans, category_of_customers = "Upcountry")])


        for _, order in orders.iterrows():
            if order['invoice_no'] not in customer_van_mapping['invoice_no'].values:
                customer_van_mapping = customer_van_mapping.append({
                    'customer_id': order['customer_id'],
                    'invoice_no': order['invoice_no'],
                    'invoice_volume': order['total_volume'],
                    'van_id': 'no van available',
                    'van_capacity': 1,
                    'customer_latitude' : order['customer_latitude'],
                    'customer_longitude' : order['customer_longitude'],
                    'customer_city' : order['customer_city']
                }, ignore_index=True)

                
        customer_van_mapping['van_id'] = customer_van_mapping['van_id'].fillna('no van available').replace('', 'no van available')    
        
        return customer_van_mapping
    
    

    #Run the van assignment
    customer_van_mapping = assign_vans(final_order_df, available_vans, max_customer, ride_batch_check)
    customer_van_mapping.drop(labels = ['customer_latitude', 'customer_longitude', 'customer_city'], axis = 1, inplace = True)
    customer_van_mapping = aggregate_metrics(customer_van_mapping, column_name='invoice_volume', type='volume', agg='sum', flag=0, percent=1)
    return customer_van_mapping



def process_ride_batch(cluster_info_df, final_order_df, van_df, ride_batch):
    available_vans = van_df.copy()
    cluster_info_df = cluster_info_df.sort_values(by='warehouse_distance', ascending=True)
    customer_van_main = pd.DataFrame(columns=['customer_id', 'invoice_no', 'invoice_volume', 'van_id', 'van_capacity', 'total_loaded_volume', 
                                              'percent_full', 'ride_batch'])

    

    for _, cluster in cluster_info_df.iterrows():
        cluster_customers = final_order_df[final_order_df['cluster'] == cluster['cluster']]
        sub_clusters = cluster_customers['sub_cluster'].unique()
        
        for sub_cluster in sub_clusters:
            sub_cluster_customers = cluster_customers[cluster_customers['sub_cluster'] == sub_cluster]

            customer_van_df = select_vans_for_cluster(sub_cluster_customers, available_vans, ride_batch_check = ride_batch, cluster_checker1 = int(cluster[0]), sub_cluster_checker1 = sub_cluster)

            # print(customer_van_df)
            # print(customer_van_df['van_id'].unique())
            # quit()

            customer_van_df['ride_batch'] = ride_batch
            customer_van_main = pd.concat([customer_van_main, customer_van_df], ignore_index=True)

            # print("\n")
            # print("Length of customer_van_main ", len(customer_van_main))
            # print("\n")

            

            customer_van_main = aggregate_metrics(customer_van_main, column_name='invoice_volume', type='volume', agg='sum', flag=0, percent=1)
            used_vans = customer_van_df['van_id'].unique()
            available_vans = available_vans[~available_vans['VehicleNumber'].isin(used_vans)]
        

    # print("length of final_order_df", len(final_order_df))
    # print("Length of cluster_info_df ", len(cluster_info_df))
    # print("Length of customer_van_main ", len(customer_van_main))
    # quit()


    final_order_df = final_order_df.merge(customer_van_main, on=['invoice_no', 'customer_id'], how='left')
    final_order_df = aggregate_metrics(final_order_df, column_name='total_quantity', type='quantity', agg='sum', flag=1, percent=0)
    final_order_df = aggregate_metrics(final_order_df, column_name='distance_from_warehouse', type='van_max_distance', agg='max', flag=1, percent=0)
    final_order_df = aggregate_metrics(final_order_df, column_name='distance_from_warehouse', type='van_min_distance', agg='min', flag=1, percent=0)
    final_order_df['is_avoid'] = ((final_order_df['van_max_distance'] >= 10) & (final_order_df['total_loaded_quantity'] <= 100)).astype(int)
    van_recommendations = final_order_df[['customer_id', 'invoice_no', 'total_quantity', 'total_volume', 'cluster', 'is_noise', 'sub_cluster', 'warehouse_location', 
                                          'warehouse_latitude', 'warehouse_longitude', 'customer_latitude', 'customer_longitude', 'distance_from_warehouse', 
                                          'van_id','ride_batch', 'van_capacity', 'total_loaded_volume', 'total_loaded_quantity', 'percent_full', 
                                          'van_max_distance','van_min_distance', 'customer_type', 'customer_city', 'zone', 'is_avoid', 'customer_order_type', 'has_multiple_orders']]

    return van_recommendations


def assign_vans_to_sub_clusters(cluster_info_df, final_order_df, van_df):
    ride_batch_1_df = process_ride_batch(cluster_info_df, final_order_df, van_df, ride_batch=1)


    no_van_orders = ride_batch_1_df[ride_batch_1_df['van_id'] == 'no van available']['invoice_no'].copy()
    ride_batch_1_df = ride_batch_1_df[ride_batch_1_df['van_id'] != 'no van available']
    
    if not no_van_orders.empty:
        final_order_df2 = final_order_df[final_order_df['invoice_no'].isin(no_van_orders)]

        ride_batch_2_df = process_ride_batch(cluster_info_df, final_order_df2, van_df, ride_batch=2)
        van_recommendations = pd.concat([ride_batch_1_df, ride_batch_2_df], ignore_index=True)

        # #addtion started here ride batch 3
        # no_van_orders2 = ride_batch_2_df[ride_batch_2_df['van_id'] == 'no van available']['invoice_no'].copy()
        # ride_batch_2_df = ride_batch_2_df[ride_batch_2_df['van_id'] != 'no van available']

        # if not no_van_orders2.empty:
        #     final_order_df3 = final_order_df[final_order_df['invoice_no'].isin(no_van_orders2)]

        #     ride_batch_3_df = process_ride_batch(cluster_info_df, final_order_df3, van_df, ride_batch=3)
        #     van_recommendations = pd.concat([ride_batch_1_df, ride_batch_2_df, ride_batch_3_df], ignore_index=True)
        # ##addtion ended here ride batch 3    
    else:
        van_recommendations = ride_batch_1_df
    

    return van_recommendations


def plot_clusters_with_customers(van_recommendations, save_path='cluster_plot_with_customers.png'):
    warehouse_lat = van_recommendations['warehouse_latitude'].iloc[0]
    warehouse_lon = van_recommendations['warehouse_longitude'].iloc[0]
    num_clusters = van_recommendations['cluster'].nunique()
    palette = sns.color_palette("tab10", num_clusters)
    palette_2 = sns.color_palette("bright", num_clusters)
    palette.extend(color for color in palette_2 if color not in palette)
    
    plt.figure(figsize=(20, 10))
    plt.scatter(warehouse_lon, warehouse_lat, color='black', label='Warehouse', s=100, marker='X')
    
    for cluster, group in van_recommendations.groupby('cluster'):
        plt.scatter(group['customer_longitude'], group['customer_latitude'], 
                    color=palette[cluster], s=100, alpha=0.6, 
                    label=f'Cluster {cluster} | Customers: {group["customer_id"].nunique()} | Orders: {group["invoice_no"].nunique()} | Vans: {group["van_id"].nunique()}')
        
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Cluster Plot')
    plt.legend(loc='best')
    plt.savefig(save_path, dpi=300)
    plt.show()
    
    print(f'Plot saved to {save_path}')
    

def calculate_van_route(van_recommendations):
    warehouse = (van_recommendations['warehouse_latitude'].iloc[0], van_recommendations['warehouse_longitude'].iloc[0])


    def find_nearest_neighbor(current_location, customers):
        route = []
        distances_covered = []
        while len(customers) > 0:
            print(customers.shape)
            distances = customers.apply(lambda row: calculate_pathway_distance(current_location, another_parameter1 = row['customer_latitude'], another_parameter2 = row['customer_longitude']), axis=1)
            # print(len(customers))
            # quit()
            nearest_index = distances.idxmin()
            route.append(nearest_index)
            distance_to_nearest = distances.loc[nearest_index]
            distances_covered.append(distance_to_nearest)
            current_location = (customers.loc[nearest_index, 'customer_latitude'], customers.loc[nearest_index, 'customer_longitude'])
            customers = customers.drop(nearest_index)
        return route, distances_covered, current_location  # Return current_location as the last customer location

  


    drop_off_orders = []
    drop_off_distances_covered = []
    
    # Iterate over each van and ride batch in the dataset
    for (van_id, ride_batch), group in van_recommendations.groupby(['van_id', 'ride_batch']):
        warehouse_coords = (group['warehouse_latitude'].iloc[0], group['warehouse_longitude'].iloc[0])
        print("In here 1")
        # Separate OKAISHIE customers and others
        okaishie_customers = group[group['customer_city'] == 'OKAISHIE'][['customer_id', 'customer_latitude', 'customer_longitude']].copy()
        other_customers = group[group['customer_city'] != 'OKAISHIE'][['customer_id', 'customer_latitude', 'customer_longitude']].copy()

        print("In here 2")
        # Route for OKAISHIE customers first
        optimal_route_indices_okaishie, distances_covered_okaishie, last_location = find_nearest_neighbor(warehouse_coords, okaishie_customers)
        
        print("In here 3")
        # Route for remaining customers
        if okaishie_customers.empty:  # Set last location to warehouse if no OKAISHIE customers
            last_location = warehouse_coords
        optimal_route_indices_other, distances_covered_other, _ = find_nearest_neighbor(last_location, other_customers)

        print("In here 4")
        # Combine both routes and distances
        combined_route_indices = optimal_route_indices_okaishie + optimal_route_indices_other
        combined_distances_covered = distances_covered_okaishie + distances_covered_other

        # Assign drop-off orders and distances
        for drop_order, (index, distance) in enumerate(zip(combined_route_indices, combined_distances_covered), start=1):
            drop_off_orders.append((index, drop_order))
            drop_off_distances_covered.append((index, distance))

        print("In here 5")
    # Convert drop-off orders and distances to DataFrames and merge them into van_recommendations
    drop_off_df = pd.DataFrame(drop_off_orders, columns=['index', 'drop_off_order']).set_index('index')
    distance_df = pd.DataFrame(drop_off_distances_covered, columns=['index', 'drop_off_distance_covered']).set_index('index')
    
    van_recommendations['drop_off_order'] = drop_off_df['drop_off_order']
    van_recommendations['drop_off_distance_covered'] = distance_df['drop_off_distance_covered']
    
    # Calculate total distance for each (van_id, ride_batch) and add it as a column
    van_recommendations['total_distance_covered'] = van_recommendations.groupby(['van_id', 'ride_batch'])['drop_off_distance_covered'].transform('sum')
    
    print("Checkpoint 18")    

    return van_recommendations
    
    
def add_fulfillment_status(van_recommendations):
    unfiltered_fulfillment_df = pd.read_csv(f"C:/Users/ME/OneDrive/Van-Route Optimization/{date}/Unfiltered Melted Orders {date}.csv")

    unfiltered_fulfillment_df = unfiltered_fulfillment_df[unfiltered_fulfillment_df["Status"].str.title() == "Approved"]
    
    unfiltered_fulfillment_df_transformed, unfiltered_van_df = transform_data(unfiltered_fulfillment_df, products_csv, warehouse_csv, vans_csv, customer_master_csv, flag = 1)

    grouped_unfiltered_fulfillment_df = pd.DataFrame(unfiltered_fulfillment_df_transformed.groupby("invoice_no").agg(
        Initial_Quantity = ('total_quantity', 'sum'),
        Initial_ProductCount = ('product_count', 'sum')
    )).reset_index()


    #merging with unfiltered fulfillment (order placement)
    van_recommendations = pd.merge(left = van_recommendations, right = grouped_unfiltered_fulfillment_df, left_on = "invoice_no", right_on = "invoice_no", how = "left", suffixes = ('', '_new'))

    customers_without_quantity = list(van_recommendations[~van_recommendations["Initial_Quantity"].notna()]["customer_id"])

    unfiltered_fulfillment_df_cust_no_qty = pd.DataFrame(unfiltered_fulfillment_df_transformed.groupby("customer_id").agg(
        Initial_Quantity = ('total_quantity', 'sum'),
        Initial_ProductCount = ('product_count', 'sum')
    )).reset_index()

    unfiltered_fulfillment_df_cust_no_qty = unfiltered_fulfillment_df_cust_no_qty[unfiltered_fulfillment_df_cust_no_qty["customer_id"].isin(customers_without_quantity)]


    van_recommendations_cust_with_qty = van_recommendations[van_recommendations["Initial_Quantity"].notna()]
    van_recommendations_cust_without_qty = van_recommendations[~van_recommendations["Initial_Quantity"].notna()]

    van_recommendations_cust_without_qty = pd.merge(left = van_recommendations_cust_without_qty, right = unfiltered_fulfillment_df_cust_no_qty, left_on = "customer_id", right_on = "customer_id", how = "left", suffixes = ('', '_new'))

    van_recommendations = pd.concat([van_recommendations_cust_with_qty, van_recommendations_cust_without_qty], ignore_index = True)

    van_recommendations["Initial_Quantity"] = np.where(~van_recommendations["Initial_Quantity"].notna(),
                                                       van_recommendations["Initial_Quantity_new"], 
                                                       van_recommendations["Initial_Quantity"])
    
    van_recommendations["Initial_ProductCount"] = np.where(~van_recommendations["Initial_ProductCount"].notna(),
                                                       van_recommendations["Initial_ProductCount_new"], 
                                                       van_recommendations["Initial_ProductCount"])
    
    van_recommendations.drop(labels = ["Initial_ProductCount_new", "Initial_Quantity_new"], axis = 1, inplace = True)

    # For the fulfilled section
    filtered_fulfillment_df = final_order_df #melted_orders_df
 
    grouped_filtered_fulfillment_df = pd.DataFrame(filtered_fulfillment_df.groupby("invoice_no").agg(
        Fulfilled_Quantity = ('total_quantity', 'sum'),
        Fulfilled_ProductCount = ('product_count', 'sum')
    )).reset_index()

    #merging with filtered fulfillment (actual delivery)
    van_recommendations = pd.merge(left = van_recommendations, right = grouped_filtered_fulfillment_df, left_on = "invoice_no", right_on = "invoice_no", how = "left", suffixes = ('', '_new'))

    
    # To get customer info added to the final output
    all_customers_wanted_columns2 = pd.DataFrame(all_customers[["Contact Code", "IsPriorityCustomer", "Organization Name"]])
    all_customers_wanted_columns2.rename(columns = {"Organization Name" : "Customer Name"}, inplace = True)
    van_recommendations = pd.merge(left = van_recommendations, right = all_customers_wanted_columns2, left_on = "customer_id", right_on = "Contact Code", how = "left", suffixes = ('', '_new'))

    van_recommendations.drop(labels = ["Contact Code"], axis = 1,  inplace = True)

    
    ## UPDATES BEGIN HERE!!
    #To cater for is_avoid cases
    van_recommendations['van_capacity'] = np.where(
    van_recommendations['is_avoid'] == 1,
    1,
    van_recommendations['van_capacity']
    )

    #To include the products breakdown of the invoice in the main output plan
    final_order_df_product_breakdown = final_order_df[['invoice_no', 'merged_products_and_quantity']]
    
    van_recommendations = pd.merge(left = van_recommendations, right = final_order_df_product_breakdown, on = 'invoice_no', how = 'left')

    ##The section that transforms the product listing
    #Split the merged_products_and_quantity into multiple columns
    split_entries = van_recommendations['merged_products_and_quantity'].str.split(',')

    #Initialize a new DataFrame to store quantities for each unique product
    expanded_df = pd.DataFrame()

    for i, row in enumerate(split_entries):
        #Dictionary to hold product quantities for each row
        product_dict = {}
        for entry in row:
            product, quantity = entry.split(':')  
            product = product.strip()  
            quantity = int(float(quantity.strip())) 
            product_dict[product] = quantity
        expanded_df = pd.concat([expanded_df, pd.DataFrame([product_dict])], ignore_index=True)

    #Fill any missing product columns with 0
    expanded_df = expanded_df.fillna(0).astype(int)

    #Merge the new columns back with the original DataFrame if needed
    van_recommendations = pd.concat([van_recommendations, expanded_df], axis=1)


    # Fulfillment Status Tag
    van_recommendations['Fulfillment_Status'] = np.where(
    van_recommendations['Fulfilled_Quantity'] == van_recommendations['Initial_Quantity'],
    'Complete Fulfillment',
    'Partial Fulfillment'
    )

    van_recommendations.loc[van_recommendations["van_capacity"] == 1, "Fulfillment_Status"] = "No Van To Fulfill"

    #No fulfillment reason/remark
    van_recommendations['Fulfillment_Remark'] = "Condidtions met and sufficient stock"

    van_recommendations['Fulfillment_Remark'] = np.where(
    (van_recommendations['zone'].str.title() == "Accra") & (van_recommendations['total_volume'] >= 30000000),
    'Big vans to Accra are not allowed',
    van_recommendations['Fulfillment_Remark']
    )

    van_recommendations['Fulfillment_Remark'] = np.where(
    (van_recommendations['zone'].str.title() == "Accra") & (van_recommendations['van_capacity'] == 1) & (((van_recommendations['total_quantity'] < 100) & (van_recommendations['distance_from_warehouse'] < 10)) | ((van_recommendations['total_quantity'] > 100))),
    'No Accra van is big enough',
    van_recommendations['Fulfillment_Remark']
    )

    van_recommendations['is_avoid'] = np.where(
    (van_recommendations['total_quantity'] <= 100) & (van_recommendations['van_id'] == 'No Van') & (
        van_recommendations['distance_from_warehouse'] >= 10
    ),
    1,
    van_recommendations['is_avoid']
    )

    van_recommendations['drop_off_distance_covered'] = np.where(
    van_recommendations['is_avoid'] == 1,
    0,
    van_recommendations['drop_off_distance_covered']
    )

    van_recommendations['total_distance_covered'] = np.where(
    van_recommendations['is_avoid'] == 1,
    0,
    van_recommendations['total_distance_covered']
    )

    van_recommendations['Fulfillment_Remark'] = np.where(
    van_recommendations['is_avoid'] == 1,
    'Distant drop off with little quantity',
    van_recommendations['Fulfillment_Remark']
    )

    #Partial fulfillment remark
    van_recommendations['Fulfillment_Remark'] = np.where(
    van_recommendations['Fulfillment_Status'] == 'Partial Fulfillment',
    'Insufficient Stock',
    van_recommendations['Fulfillment_Remark']
    )

    #Orders that can't be fulfilled by any van
    van_recommendations['Fulfillment_Remark'] = np.where(
    (van_recommendations['customer_order_type'] == 'exceeding van limit') & (van_recommendations['Fulfillment_Status'] == 'Complete Fulfillment'),
    'Condidtions met and sufficient stock but splitted order',
    van_recommendations['Fulfillment_Remark']
    )

    van_recommendations['Fulfillment_Remark'] = np.where(
    (van_recommendations['customer_order_type'] == 'exceeding van limit') & (van_recommendations['Fulfillment_Status'] == 'Partial Fulfillment'),
    'Insufficient stock but splitted order',
    van_recommendations['Fulfillment_Remark']
    )

    van_recommendations['Fulfillment_Remark'] = np.where(
    van_recommendations['van_id'] == 'no van available',
    'Ride batch 3 is required',
    van_recommendations['Fulfillment_Remark']
    )

    van_recommendations['ride_batch'] = np.where(
    (van_recommendations['van_id'] == 'no van available') | (van_recommendations['van_id'] == 'No Van'),
    0,
    van_recommendations['ride_batch']
    )

    #for cases with no vans, ensure we don't have loading kpis
    van_recommendations.loc[van_recommendations["van_capacity"] == 1, "total_loaded_quantity"] = 0
    van_recommendations.loc[van_recommendations["van_capacity"] == 1, "percent_full"] = 0
    van_recommendations.loc[van_recommendations["van_capacity"] == 1, "total_loaded_volume"] = 0
    van_recommendations.loc[van_recommendations["van_capacity"] == 1, "van_max_distance"] = 0
    van_recommendations.loc[van_recommendations["van_capacity"] == 1, "van_min_distance"] = 0
    van_recommendations.loc[van_recommendations["van_capacity"] == 1, "drop_off_order"] = 0
    van_recommendations.loc[van_recommendations["van_capacity"] == 1, "Fulfilled_Quantity"] = 0
    van_recommendations.loc[van_recommendations["van_capacity"] == 1, "Fulfilled_ProductCount"] = 0


    return van_recommendations




    
pd.set_option('display.max_columns', None)
stock_csv = f"C:/Users/ME/OneDrive/Van-Route Optimization/{date}/Stock Bal {date}.csv"
failed_stock_csv = f"C:/Users/ME/OneDrive/Van-Route Optimization/{date}/Failed Stock Check {date}.csv"
orders_csv = f"C:/Users/ME/OneDrive/Van-Route Optimization/{date}/Pending So {date}.csv"
products_csv = f"C:/Users/ME/OneDrive/Van-Route Optimization/{date}/Carton Size {date}.csv"
warehouse_csv = f"C:/Users/ME/OneDrive/Van-Route Optimization/{date}/Warehouse Details {date}.csv"
vans_csv = f"C:/Users/ME/OneDrive/Van-Route Optimization/{date}/Vehicle Volume {date}.csv"
cluster_plot_png = f"C:/Users/ME/OneDrive/Van-Route Optimization/{date}/cluster_plot {date}.png" 
van_recommendation_csv = f"C:/Users/ME/OneDrive/Van-Route Optimization/{date}/Van_Recommendation {date}.csv"
orders_df, melted_orders_df = load_and_process_orders_csv(orders_csv)

orders_df.dropna(axis = 0, how = "all", inplace = True)
melted_orders_df.dropna(axis = 0, how = "all", inplace = True)


final_order_df, van_df = transform_data(melted_orders_df, products_csv, warehouse_csv, vans_csv, customer_master_csv)

final_order_df.dropna(axis = 0, how = "all", inplace = True)
final_order_df.dropna(axis = 1, how = "all", inplace = True)

# print(final_order_df['customer_latitude'].unique())
# quit()

try:
    final_order_df.drop(labels = 0, axis = "columns", inplace = True)
except:
    pass

final_order_df.to_csv(f"C:/Users/ME/OneDrive/Van-Route Optimization/{date}/Final Orders {date}.csv", index = False)

final_order_df = get_clusters(final_order_df, eps_km=5, min_samples=4)
final_order_df = create_sub_clusters(final_order_df)
cluster_info_df = calculate_cluster_volume_weight(final_order_df)
print("Final order df", len(final_order_df))

van_recommendations = assign_vans_to_sub_clusters(cluster_info_df, final_order_df, van_df)

print("Length of van recommendations", len(van_recommendations))
 
van_recommendations = calculate_van_route(van_recommendations)

van_recommendations = add_fulfillment_status(van_recommendations)


van_recommendations.to_csv(van_recommendation_csv, index = False)




