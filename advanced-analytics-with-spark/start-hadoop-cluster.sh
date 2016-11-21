#!/usr/bin/env bash
source ~/.bash_profile
source ~/.bashrc

# Refer to HDFS Documentation to run HDFS and yarn
$HADOOP_HOME/sbin/stop-dfs.sh
$HADOOP_HOME/sbin/stop-yarn.sh

$HADOOP_HOME/sbin/start-dfs.sh
$HADOOP_HOME/sbin/start-yarn.sh