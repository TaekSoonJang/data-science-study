import org.apache.spark.util.StatCounter
import org.apache.spark.rdd.RDD

// For usage in RDD, classes should be extended as Serializable class
class NAStatCounter extends Serializable {
	val stats: org.apache.spark.util.StatCounter = new org.apache.spark.util.StatCounter()
	var missing: Long = 0

	def add(x: Double): NAStatCounter = {
		if (java.lang.Double.isNaN(x)) {
			missing += 1
		} else {
			stats.merge(x)
		}
		this
	}

	def merge(other: NAStatCounter): NAStatCounter = {
		stats.merge(other.stats)
		missing += other.missing
		this
	}

	override def toString = {
		"stats: " + stats.toString + " NaN: " + missing
	}
}

object NAStatCounter extends Serializable {
    def apply(x: Double) = new NAStatCounter().add(x)
}

// .mapPartitions is faster than .map
// http://stackoverflow.com/questions/21185092/apache-spark-map-vs-mappartitions
def statsWithMissing(rdd: RDD[Array[Double]]): Array[NAStatCounter] = {
  val nastats = rdd.mapPartitions((iter: Iterator[Array[Double]]) => {
    val nas: Array[NAStatCounter] = iter.next().map(d => NAStatCounter(d))
    iter.foreach(arr => {
      nas.zip(arr).foreach { case (n, d) => n.add(d) }
    })
    Iterator(nas)
  })

  nastats.reduce((n1, n2) => {
    n1.zip(n2).map { case (a, b) => a.merge(b) }
  })
}