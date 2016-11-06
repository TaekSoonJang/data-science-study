val rawData = sc.textFile("hdfs:///user/ds/kddcup.data")

rawData.map(_.split(',').last).countByValue().toSeq.
sortBy(_._2).reverse.foreach(println)

// remove nonnumeric value because K-Means only accepts numeric values
import org.apache.spark.mllib.linalg._

val labelsAndData = rawData.map { line =>
  val buffer = line.split(',').toBuffer
  // Remove categorical value - later these can be handled
  buffer.remove(1, 3)
  val label = buffer.remove(buffer.length - 1)
  val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
  (label, vector)
}

val data = labelsAndData.values.cache()

/*
    Now K-Means begins.
*/

import org.apache.spark.mllib.clustering._

val kmeans = new KMeans()
// By default, K=2
val model = kmeans.run(data)

model.clusterCenters.foreach(println)

val clusterLabelCount = labelsAndData.map { case (label, datum) =>
  val cluster = model.predict(datum)
  (cluster, label)
}.countByValue()

// 2 clusters don't work because only one row belongs to the second, others belong to the first.
clusterLabelCount.toSeq.sorted.foreach {
  case ((cluster, label), count) =>
    println(f"$cluster%1s$label%18s$count%8s")
}

// Euclidean distance
def distance(a: Vector, b: Vector) = 
  math.sqrt(a.toArray.zip(b.toArray).map(p => p._1 - p._2).map(d => d * d).sum)

def distToCentroid(datum: Vector, model: KMeansModel) = {
  val cluster = model.predict(datum)
  val centroid = model.clusterCenters(cluster)
  distance(centroid, datum)
}

import org.apache.spark.rdd._

def clusteringScore(data: RDD[Vector], k: Int): Double = {
  val kmeans = new KMeans()
  kmeans.setK(k)
  kmeans.setEpsilon(1.0e-6)
  val model = kmeans.run(data)
  data.map(datum => distToCentroid(datum, model)).mean()
}

(30 to 100 by 10).par.map(k => (k, clusteringScore(data, k))).toList.foreach(println)

/*
    Drawing 3-D plot of clusters, it turns out that only two features work because those values vary a lot, while others vary little.
    To complement this, all values should be normalized.
*/

def buildNormalizationFunction(data: RDD[Vector]): (Vector => Vector) = {
  val dataAsArray = data.map(_.toArray)
  val numCols = dataAsArray.first().length
  val n = dataAsArray.count()
  val sums = dataAsArray.reduce(
    (a, b) => a.zip(b).map(t => t._1 + t._2))
  val sumSquares = dataAsArray.aggregate(
    new Array[Double](numCols)
    )(
      (a, b) => a.zip(b).map(t => t._1 + t._2 * t._2),  // SeqOP
      (a, b) => a.zip(b).map(t => t._1 + t._2)          // CombineOP
    )
  val stdevs = sumSquares.zip(sums).map {
    // This is how standard devation is calculated without mean
    case (sumSq, sum) => math.sqrt(n * sumSq - sum * sum) / n
  }

  val means = sums.map(_ / n)
  (datum: Vector) => {
    val normalizedArray = (datum.toArray, means, stdevs).zipped.map(
      (value, mean, stdev) => if (stdev <= 0) (value - mean) else (value - mean) / stdev
    )
    Vectors.dense(normalizedArray)  
  }
}

val data = rawData.map { line =>
  val buffer = line.split(',').toBuffer
  buffer.remove(1, 3)
  val label = buffer.remove(buffer.length - 1)
  Vectors.dense(buffer.map(_.toDouble).toArray)
}

val normalizedData = data.map(buildNormalizationFunction(data)).cache()
(60 to 120 by 10).par.map(k => 
  (k, clusteringScore(normalizedData, k))).toList.foreach(println)

/*
    For categorical data, it can be splitted into multiple numeric data.
    Ex> TcpStates : TCP or UDP or ICMP => is_TCP / is_UDP / is_ICMP
*/

def buildCategoricalAndLabelFunction(rawData: RDD[String]): (String => (String, Vector)) = {
  val splitData = rawData.map(_.split(','))
  val protocols = splitData.map(_(1)).distinct().collect().zipWithIndex.toMap
  val services = splitData.map(_(2)).distinct().collect().zipWithIndex.toMap
  val tcpStates = splitData.map(_(3)).distinct().collect().zipWithIndex.toMap
  (line: String) => {
    val buffer = line.split(',').toBuffer
    val protocol = buffer.remove(1)
    val service = buffer.remove(1)
    val tcpState = buffer.remove(1)
    val label = buffer.remove(buffer.length - 1)
    val vector = buffer.map(_.toDouble)

    val newProtocolFeatures = new Array[Double](protocols.size)
    newProtocolFeatures(protocols(protocol)) = 1.0
    val newServiceFeatures = new Array[Double](services.size)
    newServiceFeatures(services(service)) = 1.0
    val newTcpStateFeatures = new Array[Double](tcpStates.size)
    newTcpStateFeatures(tcpStates(tcpState)) = 1.0

    vector.insertAll(1, newTcpStateFeatures)
    vector.insertAll(1, newServiceFeatures)
    vector.insertAll(1, newProtocolFeatures)

    (label, Vectors.dense(vector.toArray))
  }
}

val parseFunction = buildCategoricalAndLabelFunction(rawData)
val data = rawData.map(parseFunction).values
val normalizedData = data.map(buildNormalizationFunction(data)).cache()

(80 to 160 by 10).map(k =>
  (k, clusteringScore(normalizedData, k))).toList.foreach(println)


/*
    Sanity check by using entropy
    (http://stackoverflow.com/questions/1859554/what-is-entropy-and-information-gain)
*/

def entropy(counts: Iterable[Int]) = {
  val values = counts.filter(_ > 0)
  val n: Double = values.sum
  values.map { v =>
    val p = v / n
    -p * math.log(p)
  }.sum
}

def clusteringScoreBasedOnEntropy(normalizedLabelsAndData: RDD[(String, Vector)], k: Int) = {
  val kmeans = new KMeans()
  kmeans.setK(k)
  kmeans.setEpsilon(1.0e-6)

  val model = kmeans.run(normalizedData.values)
  val labelsAndClusters = normalizedData.mapValues(model.predict)
  val clustersAndLabels = labelsAndClusters.map(_.swap)
  val labelsInCluster = clustersAndLabels.groupByKey().values
  val labelCounts = labelsInCluster.map(
    _.groupBy(l => l).map(_._2.size))

  val n = normalizedData.count()
  labelCounts.map(m => m.sum * entropy(m)).sum / n
}

(80 to 160 by 10).map(k =>
  (k, clusteringScoreBasedOnEntropy(normalizedData, k))).toList.foreach(println)

/*
    So far, we finally found out k=150 works best for this data.
    Let's detect anomalies.
*/

def buildAnomalyDetector(
    data: RDD[Vector],
    normalizeFunction: (Vector => Vector)): (Vector => Boolean) = {
  val normalizedData = data.map(normalizeFunction)
  normalizedData.cache()

  val kmeans = new KMeans()
  kmeans.setK(150)
  kmeans.setEpsilon(1.0e-6)
  val model = kmeans.run(normalizedData)

  normalizedData.unpersist()

  val distances = normalizedData.map(datum => distToCentroid(datum, model))
  val threshold = distances.top(100).last

  (datum: Vector) => distToCentroid(normalizeFunction(datum), model) > threshold
}

val originalAndData = rawData.map(line => (line, parseFunction(line)._2))
val data = originalAndData.values
val normalizeFunction = buildNormalizationFunction(data)
val anomalyDetector = buildAnomalyDetector(data, normalizeFunction)
val anomalies = originalAndData.filter {
  case(original, datum) => anomalyDetector(datum)
}.keys
anomalies.take(10).foreach(println)







