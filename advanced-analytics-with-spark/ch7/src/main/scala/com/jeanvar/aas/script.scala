import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import edu.umd.cloud9.collection.XMLInputFormat
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.{Text, LongWritable}


def loadMedline(sc: SparkContext, path: String) = {
  @transient val conf = new Configuration()
  conf.set(XMLInputFormat.START_TAG_KEY, "<MedlineCitation ")
  conf.set(XMLInputFormat.END_TAG_KEY, "</MedlineCitation>")
  val in = sc.newAPIHadoopFile(path, classOf[XMLInputFormat], classOf[LongWritable], classOf[Text], conf)
  in.map(line => line._2.toString)
}
val medline_raw = loadMedline(sc, "medline")

import scala.xml._

val cit = <MedlineCitation>data</MedlineCitation>
val raw_xml = medline_raw.take(1)(0)
val elem = XML.loadString(raw_xml)

def majorTopics(elem: Elem): Seq[String] = {
  val dn = elem \\ "DescriptorName"
  val mt = dn.filter(n => (n \ "@MajorTopicYN").text == "Y")
  mt.map(n => n.text)
}
majorTopics(elem)

val mxml: RDD[Elem] = medline_raw.map(XML.loadString)
val medline: RDD[Seq[String]] = mxml.map(majorTopics).cache()

/* Below are common stats which are not very useful for graph analysis */

// medline.take(1)(0)
// medline.count()

val topics: RDD[String] = medline.flatMap(mesh => mesh)
val topicCounts = topics.countByValue()
// topicCounts.size

val tcSeq = topicCounts.toSeq
tcSeq.sortBy(_._2).reverse.take(10).foreach(println)

val valueDist = topicCounts.groupBy(_._2).mapValues(_.size)
valueDist.toSeq.sorted.take(10).foreach(println)

val topicsPairs = medline.flatMap(t => t.sorted.combinations(2))
val coocuurs = topicsPairs.map(p => (p, 1)).reduceByKey(_+_)
coocuurs.cache()
// coocuurs.count()

// val ord = Ordering.by[(Seq[String], Int), Int](_._2)
// coocuurs.top(10)(ord).foreach(println)


import com.google.common.hash.Hashing

def hashId(str: String) = {
  Hashing.md5().hashString(str).asLong()
}

val vertices = topics.map(topic => (hashId(topic), topic))

/*  Check if hashing works well
val uniqueHashes = vertices.map(_._1).countByValue()
val uniqueTopics = vertices.map(_._2).countByValue()
uniqueHashes.size == uniqueTopics.size
*/

import org.apache.spark.graphx._

val edges = coocuurs.map(p => {
  val (topics, cnt) = p
  val ids = topics.map(hashId).sorted
  Edge(ids(0), ids(1), cnt)
})

val topicGraph = Graph(vertices, edges)
topicGraph.cache()

/*  Check redundancy has been removed by Graph API
vertices.count()
topicGraph.vertices.count()
*/

/* connected components are like islands that are connected to each other, but not with other islands */
val connectedComponentGraph: Graph[VertexId, Int] = topicGraph.connectedComponents()

def sortedConnectedComponents(connectedComponents: Graph[VertexId, _]): Seq[(VertexId, Long)] = {
  val componentCounts = connectedComponents.vertices.map(_._2).countByValue
  componentCounts.toSeq.sortBy(_._2).reverse
}

val componentCounts = sortedConnectedComponents(connectedComponentGraph)
// componentCounts.size
// componentCounts.take(10).foreach(println)

val nameCID = topicGraph.vertices.innerJoin(connectedComponentGraph.vertices) {
  (topicId, name, componentId) => (name, componentId)
}

val c1 = nameCID.filter(x => x._2._2 == componentCounts(1)._1)
// c1.collect().foreach(x => println(x._2._1))

/*  Find HIV related topics
val hiv = topics.filter(_.contains("HIV")).countByValue()
hiv.foreach(println)
*/

/* Degree is the number of edges connected to a vertex */
val degrees: VertexRDD[Int] = topicGraph.degrees.cache()
// degrees.map(_._2).stats()

def topNamesAndDegrees(degrees: VertexRDD[Int], topicGraph: Graph[String, Int]): Array[(String, Int)] = {
  val namesAndDegrees = degrees.innerJoin(topicGraph.vertices) {
    (topicId, degree, name) => (name, degree)
  }
  val ord = Ordering.by[(String, Int), Int](_._2)
  namesAndDegrees.map(_._2).top(10)(ord)
}

// topNamesAndDegrees(degrees, topicGraph).foreach(println)

/*
    Chi-Square to filter low-relevance relationship between topics
*/

def chiSq(YY: Int, YB: Int, YA: Int, T:Long): Double = {
  val NB = T - YB
  val NA = T - YA
  val YN = YA - YY
  val NY = YB - YY
  val NN = T - NY - YN - YY
  val inner = math.abs(YY * NN - YN * NY) - T / 2.0
  T * math.pow(inner, 2) / (YA * NA * YB * NB)
}

val T = medline.count()
val topicCountsRdd = topics.map(x => (hashId(x), 1)).reduceByKey(_+_)
val topicCountGraph = Graph(topicCountsRdd, topicGraph.edges)

val chiSquaredGraph = topicCountGraph.mapTriplets(triplet => {
  chiSq(triplet.attr, triplet.srcAttr, triplet.dstAttr, T)  
})

// chiSquaredGraph.edges.map(x => x.attr).stats()

/* Referring to stats, stdev is very large, so we can apply aggresive filtering like 99.999 (in chi sq with degree of freedom 1, its value 19.5) */
val interesting = chiSquaredGraph.subgraph(
  triplet => triplet.attr > 19.5)
// interesting.edges.count

val interestingComponentCounts = sortedConnectedComponents(interesting.connectedComponents())
val interestingDegrees = interesting.degrees.cache()
interestingDegrees.map(_._2).stats()

/*
    'small-world' network - represented as mostly realistic graph characteristics
    Clique - fully connected subgraph, which is NP-complete problem to find it perfectly
    Triangle Count - Connected graph wih three vertices
    Local Clustering Coefficient - Referring local density of graph
*/

// It fires bug in Spark 2.0 and works fine in Spark 1.6
// (http://stackoverflow.com/questions/40337366/spark-graphx-requirement-failed-invalid-initial-capacity)
val triCountGraph = interesting.triangleCount()
interesting.vertices.map(x => x._2).stats()

val maxTrisGraph = interesting.degrees.mapValues(d => d * (d - 1) / 2.0)
val clusterCoefGraph = triCountGraph.vertices.
  innerJoin(maxTrisGraph) { (vertexId, triCount, maxTris) => {
    if (maxTris == 0) 0 else triCount / maxTris
  }
}

clusterCoefGraph.map(_._2).sum()/ interesting.vertices.count()

/*
    Using pregel to find out average distance in graph
*/

def mergeMaps(m1: Map[VertexId, Int], m2: Map[VertexId, Int]): Map[VertexId, Int] = {
  def minThatExists(k: VertexId): Int = {
    math.min(m1.getOrElse(k, Int.MaxValue), m2.getOrElse(k, Int.MaxValue))
  }

  (m1. keySet ++ m2.keySet).map {
    k => (k, minThatExists(k))
  }.toMap
}

def update(id: VertexId, state: Map[VertexId, Int], msg: Map[VertexId, Int]) = {
  mergeMaps(state, msg)
}

def checkIncrement(a: Map[VertexId, Int], b: Map[VertexId, Int], bid: VertexId) = {
  val aplus = a.map { case (v, d) => v -> (d + 1) }
  if (b != mergeMaps(aplus, b)) {
    Iterator((bid, aplus))
  } else {
    Iterator.empty
  }
}

def iterate(e: EdgeTriplet[Map[VertexId, Int], _]) = {
  checkIncrement(e.srcAttr, e.dstAttr, e.dstId) ++
  checkIncrement(e.dstAttr, e.srcAttr, e.srcId)
}

val fraction = 0.02
val replacement = false // extraction without replacement
val sample = interesting.vertices.map(v => v._1).sample(replacement, fraction, 1729L)
val ids = sample.collect().toSet

val mapGraph = interesting.mapVertices((id, _) => {
  if (ids.contains(id)) {
    Map(id -> 0)
  } else {
    Map[VertexId, Int]()
  }
})

val start = Map[VertexId, Int]()
val res = mapGraph.pregel(start)(update, iterate, mergeMaps)

val paths = res.vertices.flatMap { case (id, m) =>
  m.map { case (k, v) =>
    if (id < k) {
      (id, k, v)
    } else {
      (k, id, v)
    }
  }
}.distinct()
paths.cache()

paths.map(_._3).filter(_ > 0).stats()

val hist = paths.map(_._3).countByValue()
hist.toSeq.sorted.foreach(println)




















