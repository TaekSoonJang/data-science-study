import scala.util.Random
import scala.collection.Map

import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.recommendation._

def areaUnderCurve(
    positiveData: RDD[Rating],
    bAllItemIDs: Broadcast[Array[Int]],
    predictFunction: (RDD[(Int, Int)] => RDD[Rating])) = {
  // What this actually computes is AUC, per user. The result is actually something
  // that might be called "mean AUC".

  val positiveUserProducts = positiveData.map(r => (r.user, r.product))

  // True Positive
  val positivePredictions = predictFunction(positiveUserProducts).groupBy(_.user)

  val negativeUserProducts = positiveUserProducts.groupByKey().mapPartitions {
    userIDAndPosItemIDs => {
      val random = new Random()
      val allItemIDs = bAllItemIDs.value
      userIDAndPosItemIDs.map { case (userID, posItemIDs) =>
        val posItemIDSet = posItemIDs.toSet
        val negative = new scala.collection.mutable.ArrayBuffer[Int]()
        var i = 0
        while (i < allItemIDs.size && negative.size < posItemIDSet.size) {
          val itemID = allItemIDs(random.nextInt(allItemIDs.size))
          if (!posItemIDSet.contains(itemID)) {
            negative += itemID
          }
          i += 1
        }

        negative.map(itemID => (userID, itemID))
      }
    }
  }.flatMap(t => t)

  // False Positive
  val negativePredictions = predictFunction(negativeUserProducts).groupBy(_.user)

  positivePredictions.join(negativePredictions).values.map {
    case (positiveRatins, negativeRatings) =>
      var correct = 0L
      var total = 0L
      for (positive <- positiveRatins;
           negative <- negativeRatings) {
        if (positive.rating > negative.rating) {
          correct += 1
        }

        total += 1
      }

      correct.toDouble / total
  }.mean()
}

def buildRatings(
    rawUserArtistData: RDD[String],
    bArtistAlias: Broadcast[Map[Int, Int]]) = {
  rawUserArtistData.map { line =>
      val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
      val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
      Rating(userID, finalArtistID, count)
  }
}

// Simple prediction
def predictMostListened(
    sc: SparkContext,
    train: RDD[Rating])(allData: RDD[(Int, Int)]) = {
  val bListenCount = sc.broadcast(
    train.map(r => (r.product, r.rating))
    .reduceByKey(_ + _).collectAsMap()
  )
  allData.map { case (user, product) =>
    Rating(
      user,
      product,
      bListenCount.value.getOrElse(product, 0.0)
    )
  }
}





