#!/usr/bin/env bash

# Refer to HDFS Documentation to run HDFS and yarn
sbin/start-dfs.sh
sbin/start-yarn.sh

# Run spark shell on yarn
# https://www.altiscale.com/blog/tips-and-tricks-for-running-spark-on-hadoop-part-4-memory-settings/
bin/spark-shell \
--master yarn \
--deploy-mode client \
--executor-memory 6g \
--driver-memory 6g \
--conf spark.yarn.executor.memoryOverhead=600 \
--executor-cores 6 \
--verbose