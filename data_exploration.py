## Kai Alcayde
## this is some code used to mess around and learn about how dataFrames work.

def load_housing_data(housing_path):
    return pd.read_csv("https://drive.google.com/uc?id=1l8mTUz-1D4Gq4LtEYX8IDCGl8AmoRuYj")

# load airbnb data
df = pd.read_csv('/content/sample_data/AB_NYC_2019.csv')
df.info()

# drop name, host_id, host_name, last_review, reviews_per_month
# make data more numerical
columns_to_drop = ['name','host_id','host_name','last_review','reviews_per_month']
df = df.drop(columns=columns_to_drop, axis=1)

# Make box plots, create separate subplots to get a sense of the data
# fig makes one figure, axes is subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

# Create box plots for 'price,' 'minimum_nights,' and 'availability_365'
price = axes[0].boxplot(df['price'], labels=['Price'])
min_nights = axes[1].boxplot(df['minimum_nights'], labels=['Minimum Nights'])
avail = axes[2].boxplot(df['availability_365'], labels=['Availability 365'])

# Show the box plots
plt.show()

# plot average price of listing per neighbourhood grouo
df['neighbourhood_group'].value_counts()
# split data based on (neighbourhood_group feature), then sort by [price] average
df_groups = df.groupby("neighbourhood_group")[["price"]].mean()
df_groups = df_groups.sort_values(by='price', ascending=True)

# visualize average price by neighbourhood group in NYC
# AKA Queens, etc
plt.plot(df_groups)
plt.xlabel('city')
plt.ylabel('price')
plt.title('Average price by Neighbourhood Group')

# visualize average price as histograms across all neighbourhood groups
df_bronx = df.groupby(["neighbourhood_group"]).get_group("Bronx")['price']
df_queens = df.groupby(["neighbourhood_group"]).get_group("Queens")['price']
df_staten = df.groupby(["neighbourhood_group"]).get_group("Staten Island")['price']
df_brooklyn = df.groupby(["neighbourhood_group"]).get_group("Brooklyn")['price']
df_manhattan = df.groupby(["neighbourhood_group"]).get_group("Manhattan")['price']

neighbourhoods = ['Bronx', 'Queens', 'Staten Island', 'Brooklyn', 'Manhattan']
group_prices = [df_bronx, df_queens, df_staten, df_brooklyn, df_manhattan]

# start plotting
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
num_bins =50
bins = np.linspace(0, 600, num_bins)
bins = np.append(bins, np.inf)        # for outliers


# flatten multi-dimensional array of subplot axes into a 1D array to work in for loop
for i, subplot in enumerate(axs.flat):

    # if the df exists, plot
    if i < len(group_prices):
        subplot.hist(group_prices[i], bins=bins)
        subplot.set_title(neighbourhoods[i])
        subplot.set_xlabel('Price')
        subplot.set_ylabel('Frequency')
    else:
        # else hide any empty subplots
        subplot.axis('off')

plt.show()


# look at airbnb spread throughout nyc
# df.plot(kind="scatter", x="longitude", y="latitude", alpha = 0.1)

# load an image of ny
images_path = os.path.join('./', "images")
os.makedirs(images_path, exist_ok=True)
filename = "nyc.png"

import matplotlib.image as mpimg
ny_image = mpimg.imread(os.path.join(images_path, filename))
ax = df.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7), alpha = 0.5)

# overlay the nyc map on the plotted scatter plot

# determine  geographical range of data
min_longitude = df['longitude'].min()
max_longitude = df['longitude'].max()
min_latitude = df['latitude'].min()
max_latitude = df['latitude'].max()

# overlay the NYC image based on actual range of your data
plt.imshow(ny_image, extent=[min_longitude-0.00,
                             max_longitude-0.0,
                             min_latitude-0.0,
                             max_latitude+0.0],
                             alpha=0.5)
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

plt.legend(fontsize=16)
plt.title("AirBnB's in NYC")
save_fig("nyc_housing_prices_plot")
plt.show()


# find average price of room types in Manhattan, where available for more than 180 days
# filter rows where availability_365 > 180
# nested [] is important as it gets those values, in excahnge for just having a T/F list
df_sub = df[df["availability_365"] > 180]

# filter for the "Manhattan" neighborhood group
df_manhattan = df_sub[df_sub["neighbourhood_group"] == "Manhattan"]

# get average prices only
df_manhattan = df_manhattan.groupby('room_type')['price'].mean()

# in this case, we did not do sort_values by='price'. this is because this groupby
# does not have label price included(not sure why it works without). so just sort by axis
df_manhattan = df_manhattan.sort_values(axis = 0, ascending=True)

plt.plot(df_manhattan)
plt.title('Room prices in Manhattan, availability > 180 days')
plt.xlabel('Type of Room')
plt.ylabel('Average Price')
plt.show()
