val taxiRaw = sc.textFile("taxidata")
// val taxiHead = taxiRaw.take(10)
// taxiHead.foreach(println)

import java.text.SimpleDateFormat
import com.esri.core.geometry.Point
import com.github.nscala_time.time.Imports._
import com.jeanvar.aas.ch8._

val formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")

def point(longitude: String, latitude: String): Point = {
  new Point(longitude.toDouble, latitude.toDouble)
}

def parse(line: String): (String, Trip) = {
  val fields = line.split(',')
  val license = fields(1)
  val pickupTime = new DateTime(formatter.parse(fields(5)))
  val dropoffTime = new DateTime(formatter.parse(fields(6)))
  val pickupLoc = point(fields(10), fields(11))
  val dropoffLoc = point(fields(12), fields(13))

  val trip = Trip(pickupTime, dropoffTime, pickupLoc, dropoffLoc)
  (license, trip)
}

def safe[S, T](f: S => T): S => Either[T, (S, Exception)] = {
  new Function[S, Either[T, (S, Exception)]] with Serializable {
    def apply(s: S): Either[T, (S, Exception)] = {
      try {
        Left(f(s))
      } catch {
        case e: Exception => Right((s, e))
      }
    }
  }
}

val safeParse = safe(parse)
val taxiParsed = taxiRaw.map(safeParse)
taxiParsed.cache()

/*  Found out how many bad records exist in the dataset.

taxiParsed.map(_.isLeft).countByValue().foreach(println)

val taxiBad = taxiParsed.collect({
  case t if t.isRight => t.right.get
})

taxiBad.collect().foreach(println)
*/

/* Decided to exclude bad records because the number is so small */
val taxiGood = taxiParsed.collect({
  case t if t.isLeft => t.left.get
})

taxiGood.cache()

/* Look into difference between pickup time and drop off time so that unrealistic data are excluded */
import org.joda.time.Duration

def hours(trip: Trip): Long = {
  val d = new Duration(
    trip.pickupTime,
    trip.dropoffTime)
  d.getStandardHours
}

/* Turned out that most records belong to 0~3 hours */
// taxiGood.values.map(hours).countByValue().toList.sorted.foreach(println)

val taxiClean = taxiGood.filter {
  case (lic, trip) => {
    val hrs = hours(trip)
    0 <= hrs && hrs < 3
  }
}

val geojson = scala.io.Source.fromFile("/Users/taeksoonjang/SourceTreeRepos/data-science-study/advanced-analytics-with-spark/ch8/nyc-boroughs.geojson").mkString

import spray.json._
import com.jeanvar.aas.ch8.GeoJsonProtocol._

val features = geojson.parseJson.convertTo[FeatureCollection]

/*  Test features get borough names correctly.
val p = new Point(-73.994499, 40.75066)
val borough = features.find(f => f.geometry.contains(p))
*/

val areaSortedFeatures = features.sortBy(f => {
  val borough = f("boroughCode").convertTo[Int]
  (borough, -f.geometry.area2D())
})

val bFeatures = sc.broadcast(areaSortedFeatures)

def borough(trip: Trip): Option[String] = {
  val feature: Option[Feature] = bFeatures.value.find(f => {
    f.geometry.contains(trip.dropoffLoc)
  })
  feature.map(f => {
    f("borough").convertTo[String]
  })
}

// taxiClean.values.map(borough).countByValue().foreach(println)
// taxiClean.values.filter(t => borough(t).isEmpty).take(10).foreach(println)

/* Remove location is x=0, y=0 which means no geo info offered */
def hasZero(trip: Trip): Boolean = {
  val zero = new Point(0.0, 0.0)
  (zero.equals(trip.pickupLoc) || zero.equals(trip.dropoffLoc))
}

val taxiDone = taxiClean.filter {
  case (lic, trip) => !hasZero(trip)
}.cache()

// taxiDone.values.map(borough).countByValue().foreach(println)

/* Sessionization */

import org.apache.spark.rdd._
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

def split(t1: Trip, t2: Trip): Boolean = {
    val p1 = t1.pickupTime
    val p2 = t2.pickupTime
    val d = new Duration(p1, p2)
    d.getStandardHours >= 4 // 4 is arbitrarily selected value to define one work time
}

def secondaryKeyFunc(trip: Trip) = trip.pickupTime.getMillis

def groupSorted[K, V, S](
    it: Iterator[((K, S), V)],
    splitFunc: (V, V) => Boolean): Iterator[(K, List[V])] = {
  var curLic: K = null.asInstanceOf[K]
  val curTrips = ArrayBuffer[V]()
  it.flatMap { case ((lic, _), trip) =>
    if (!lic.equals(curLic) || splitFunc(curTrips.last, trip)) {
      val result = (curLic, List(curTrips:_*))
      curLic = lic
      curTrips.clear()
      curTrips += trip
      if (result._2.isEmpty) None else Some(result)
    } else {
      curTrips += trip
      None
    }
  } ++ Iterator((curLic, List(curTrips:_*)))
}

def groupByKeyAndSortValues[K : Ordering : ClassTag, V : ClassTag, S : Ordering](
    rdd: RDD[(K, V)],
    secondaryKeyFunc: (V) => S,
    splitFunc: (V, V) => Boolean): RDD[(K, List[V])] = {
  val presess = rdd.map {
    case (lic, trip) => {
      ((lic, secondaryKeyFunc(trip)), trip)
    }
  }
  val partitioner = new FirstKeyPartitioner[K, S](presess.partitions.length)
  presess.repartitionAndSortWithinPartitions(partitioner).mapPartitions(groupSorted(_, splitFunc))
}

val sessions = groupByKeyAndSortValues(taxiDone, secondaryKeyFunc, split)
sessions.cache()

def boroughDuration(t1: Trip, t2: Trip) = {
  val b = borough(t1)
  val d = new Duration(
    t1.dropoffTime,
    t2.pickupTime)
  (b, d)
}

val boroughDurations: RDD[(Option[String], Duration)] = 
  sessions.values.flatMap(trips => {
    val iter: Iterator[Seq[Trip]] = trips.sliding(2)
    val viter = iter.filter(_.size == 2)
    viter.map(p => boroughDuration(p(0), p(1)))
  }).cache()




