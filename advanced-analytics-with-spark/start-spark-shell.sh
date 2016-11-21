#!/usr/bin/env bash
source ~/.bash_profile
source ~/.bashrc

# Run spark shell on yarn
# https://www.altiscale.com/blog/tips-and-tricks-for-running-spark-on-hadoop-part-4-memory-settings/
$SPARK_HOME/bin/spark-shell \
--master yarn \
--deploy-mode client \
--executor-memory 9g \
--driver-memory 3g \
--conf spark.yarn.executor.memoryOverhead=600 \
--num-executors 1 \
--executor-cores 6 \
--verbose \
--jars /Users/taeksoonjang/SourceTreeRepos/data-science-study/advanced-analytics-with-spark/ch7/target/ch7-0.9.jar