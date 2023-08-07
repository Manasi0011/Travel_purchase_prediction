from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import pandas as pd



cloud_config = {
  'secure_connect_bundle': 'D:\Study\Internship_ML\secure-connect-tourism-dataset.zip'
}
auth_provider = PlainTextAuthProvider('EENnEssbdUJdhFpmZUYBodgF',
                                      'zAaREg+NehgMq_lebKrCflpgTILwb2Fd_Ie9bC.XvOIFW2YCpA3ji_0y+UKXMKoD6+wT40rpPsvdxZNzRD8gyrk16PP+zZ-lbD0IQM6IliENzFcF9Hkd+zlqEeZ,D9BF')
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
# cluster = Cluster(['0.0.0.0'], 9042)
session = cluster.connect()

row = session.execute("select release_version from system.local").one()
if row:
  print(row[0])
else:
  print("An error occurred.")


# Execute a SELECT statement
result_set = session.execute('SELECT * FROM db."Tourist_Dataset" LIMIT  100')

#
# # Process the result set
# for row in result_set:
#     # Access specific columns from the row
#     column1_value = row.PitchSatisfactionScore
#     column2_value = row.Age
#     # Print or use the retrieved values
#     print(f"Column1: {column1_value}, Column2: {column2_value}")


# Create an empty dictionary to store the column values
data = {}


# Process the result set and populate the dictionary
for row in result_set:
    row_dict = dict(row._asdict())
    for column_name in row_dict:
        if column_name not in data:
            data[column_name] = []
        data[column_name].append(row_dict[column_name])

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Print or use the DataFrame as needed
# print(df.head())

print(df.shape)
print(df.info())
# Close the session and cluster connection
session.shutdown()
cluster.shutdown()


