import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

#create spark session
spark = SparkSession\
    .builder\
    .master('local[2]')\
    .appName('quakes_etl')\
    .config('spark.jars.packages', 'org.momgodb.spark:mongo-spark-connector_2.12:2.4.1')\
    .getOrCreate()

df_load =spark.read.csv(r"C:\Users\91902\Downloads\database.csv", header=True)

lst_dropped_columns=['Depth Error','Time','Depth Seismic Stations','Magnitude Error','Magnitude Seismic Stations'
                    ,'Azimuthal Gap','Root Mean Square','Source','Location Source','Magnitude Source','Status']

df_load = df_load.drop(*lst_dropped_columns)

df_load = df_load.withColumn('Year', year(to_timestamp('Date', 'dd/MM/yyyy')))

df_quake_freq = df_load.groupBy('Year').count().withColumnRenamed('count', 'Counts')

df_load = df_load.withColumn('Latitude', df_load['Latitude'].cast(DoubleType()))\
    .withColumn('Longitude', df_load['Longitude'].cast(DoubleType()))\
    .withColumn('Depth', df_load['Depth'].cast(DoubleType()))\
    .withColumn('Magnitude', df_load['Magnitude'].cast(DoubleType()))


df_max = df_load.groupBy('Year').max('Magnitude').withColumnRenamed('max(Magnitude)', 'Max_Magnitude')
df_avg = df_load.groupBy('Year').max('Magnitude').withColumnRenamed('avg(Magnitude)', 'Avg_Magnitude')

df_quake_freq = df_quake_freq.join(df_avg,['Year']).join(df_max, ['Year'])

df_load.dropna()
df_quake_freq.dropna()

df_load.write.format('mongo')\
    .mode('overwrite')\
    .option('spark.mongodb.output.uri', 'mongodb://127.0.0.1:27017/Quake.quakes').save()

df_quake_freq.write.format('mongo')\
    .mode('overwrite')\
    .option('spark.mongodb.output.uri', 'mongodb://127.0.0.1:27017/Quake.quake_freq').save()

print(df_quake_freq.show(5))
print(df_load.show(5))

print("INFO: ran successfully")
print("")