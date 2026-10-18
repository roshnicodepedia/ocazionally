"""Quick check that PySpark works on your laptop."""
import os, sys
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = "C:\\hadoop\\bin;" + os.environ["PATH"]
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("ocazionally-test")
    .master("local[2]")
    .config("spark.driver.memory", "3g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

data = [
    ("V0001", "decor", "Gandhinagar", 25000),
    ("V0002", "catering", "Ahmedabad", 60000),
    ("V0003", "sound_dj", "Gandhinagar", 15000),
]
df = spark.createDataFrame(data, ["vendor_id", "category", "city", "price_min"])
df.show()
print("Row count:", df.count())
df.groupBy("city").avg("price_min").show()

df.write.mode("overwrite").parquet("data/synthetic/test_vendors.parquet")
print("Parquet write OK")
spark.stop()
