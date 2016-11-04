
// Considering size of text file is 400mb, it is generally divided into 3~6 blocks.
// However, it is better to divided in more blocks for efficient calculation with more cores.
// Refer to API docs for parameters of .textFile
val rawUserArtistData = sc.textFile("hdfs:///user/ds/user_artist_data.txt")

// rawUserArtistData.map(_.split(' ')(0).toDouble).stats()
// rawUserArtistData.map(_.split(' ')(1).toDouble).stats()

val rawArtistData = sc.textFile("hdfs:///user/ds/artist_data.txt")
val artistByID = rawArtistData.flatMap { line =>
  val (id, name) = line.span(_ != '\t')
  if (name.isEmpty) {
    None
  } else {
    try {
      Some((id.toInt, name.trim))
    } catch {
      case e: NumberFormatException => None
    }
  }
}

val rawArtistAlias = sc.textFile("hdfs:///user/ds/artist_alias.txt")
val artistAlias = rawArtistAlias.flatMap { line => 
  val tokens = line.split('\t')
  if (tokens(0).isEmpty) {
    None
  } else {
    Some((tokens(0).toInt, tokens(1).toInt))
  }
}.collectAsMap()

/*
  artistAlias will be used in .map closure and copied to all tasks.
  Considering multiple tasks run on same node, copying same object to all JVMs in the same node
    is very inefficient in terms of network traffic and memory usage.
  By broadcasting it, this object is copied only one per executor
    so same object will be shared among tasks.
*/
val bArtistAlias = sc.broadcast(artistAlias)

import org.apache.spark.mllib.recommendation._

val trainData = rawUserArtistData.map { line =>
  val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
  val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
  Rating(userID, finalArtistID, count)
}.cache()

val model = ALS.trainImplicit(
  trainData,
  10,       // rank - # of features
  5,        // iteration
  0.01,     // lambda - to reduce overfitting
  1.0       // alpha - to control weight between observed and non-observed items
)
model.userFeatures.mapValues(_.mkString(", ")).first()

val rawArtistsForUser = rawUserArtistData.map(_.split(' ')).filter { case Array(user,_,_) => user.toInt == 2093760 }
val existingProducts = rawArtistsForUser.map { case Array(_,artist,_) => artist.toInt }.collect.toSet
artistByID.filter { case (id, name) =>
  existingProducts.contains(id)
}.values.collect().foreach(println)

val recommendations = model.recommendProducts(2093760, 5)
recommendations.foreach(println)

val recommendedProductIDs = recommendations.map(_.product).toSet
artistByID.filter { case (id, name) =>
  recommendedProductIDs.contains(id)
}.values.collect().foreach(println)

/*
    Calculate AUC(accuracy of recommendation)
*/

// To use areaUnderCurve function in AUC.scala
// :load /Users/taeksoonjang/SourceTreeRepos/data-science-study/advanced-analytics-with-spark/ch3/AUC.scala

val allData = buildRatings(rawUserArtistData, bArtistAlias)
val Array(trainData, cvData) = allData.randomSplit(Array(0.9, 0.1))
trainData.cache()
cvData.cache()

val allItemIDs = allData.map(_.product).distinct().collect()
val bAllItemIDs = sc.broadcast(allItemIDs)

val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)

// val auc = areaUnderCurve(cvData, bAllItemIDs, model.predict)

// Our model should be compared to simple value to make sure it is worth making model.
// val auc = areaUnderCurve(cvData, bAllItemIDs, predictMostListened(sc, trainData))

// Optimize hyperparameters
val evaluations = 
  for (rank <- Array(10, 50);
       lambda <- Array(1,0, 0.0001);
       alpha <- Array(1.0, 40.0))
    yield {
      val model = ALS.trainImplicit(trainData, rank, 10, lambda, alpha)
      val auc = areaUnderCurve(cvData, bAllItemIDs, model.predict)
      ((rank, lambda, alpha), auc)
    }

val someUsers = allData.map(_.user).distinct().take(100)
val someRecommendations = someUsers.map(useID => model.recommendProducts(userID, 5))
someRecommendations.map(
  recs => recs.head.user + " -> " + recs.map(_.product).mkString(", ")
).foreach(println)





