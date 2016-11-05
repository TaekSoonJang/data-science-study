import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._

val rawData = sc.textFile("hdfs:///user/ds/covtype.data")

val data = rawData.map { line =>
  val values = line.split(',').map(_.toDouble)
  val featureVector = Vectors.dense(values.init)
  val label = values.last - 1
  LabeledPoint(label, featureVector) // Abstract of feature vector
}

val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))

trainData.cache()
// To optimize hyperparameter
cvData.cache()
// To evaluate perfomance or accuracy of model
testData.cache()

import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._

def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
  val predictionsAndLabels = data.map(example =>
    (model.predict(example.features), example.label)
  )
  // Evaluation class for quality of model
  // Refer to BinaryClassificationMetrics
  new MulticlassMetrics(predictionsAndLabels)
}

val model = DecisionTree.trainClassifier(
  trainData, 
  7, 
  Map[Int, Int](),  // key(identifier of categorical value), value(cardinality)
  "gini",           // impurity check method - "gini" or "entropy"
  4,                // max depth
  100               // max bin(condition in each node)
)

val metrics = getMetrics(model, cvData)
/*
    Confusion Matrix
    - row : real value
    - column : predicted value
    - (i, j) : # of count that real value i is actually predicted as j
    - (i, i) : Right prediction
    - (i, j) where i != j : Wrong prediction
*/
metrics.confusionMatrix
metrics.accuracy
metrics.precision // deprecated
metrics.recall // deprecated

(0 until 7).map(
  cat => (metrics.precision(cat), metrics.recall(cat))
).foreach(println)

// Probabilites of random categorization
// This will be used to compare results with that of our model
def classProbabilities(data: RDD[LabeledPoint]): Array[Double] = {
  val countsByCategory = data.map(_.label).countByValue()
  val counts = countsByCategory.toArray.sortBy(_._1).map(_._2)
  counts.map(_.toDouble / counts.sum)
}
val trainPriorProbabilites = classProbabilities(trainData)
val cvPriorProbabilities = classProbabilities(cvData)
trainPriorProbabilites.zip(cvPriorProbabilities).map {
  case (trainProb, cvProb) => trainProb * cvProb
}.sum

val evaluations = 
  for (impurity <- Array("gini", "entropy");
       depth <- Array(1, 20);
       bins <- Array(10, 300))
    yield {
      val model = DecisionTree.trainClassifier(
        trainData, 7, Map[Int, Int](), impurity, depth, bins)
      val predictionsAndLabels = cvData.map(example =>
        (model.predict(example.features), example.label)
      )
      val accuracy = new MulticlassMetrics(predictionsAndLabels).accuracy
      ((impurity, depth, bins), accuracy)
    }

// It turns out that ("entropy", 20 depth, 300 bins) works best.
evaluations.sortBy(_._2).reverse.foreach(println)

// After hyperparameters are set, train model with train and cv data and get accuracy from test data
val model = DecisionTree.trainClassifier(
  trainData.union(cvData), 7, Map[Int, Int](), "entropy", 20, 300)

/*
    So far, we treated categorical value as nemeric because it was encoded as 0 or 1 which can be numeric also.
    From now on, we treat categorical value as it is.
*/

val data = rawData.map { line =>
  val values = line.split(',').map(_.toDouble)
  val wilderness = values.slice(10, 14).indexOf(1.0).toDouble
  val soil = values.slice(14, 54).indexOf(1.0).toDouble
  val featureVector = Vectors.dense(values.slice(0, 10) :+ wilderness :+ soil)
  val label = values.last - 1
  LabeledPoint(label, featureVector)
}

val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))

val evaluations = 
  for (impurity <- Array("gini", "entropy");
       depth <- Array(10, 20, 30);
       // As one of categorical value's cardinality is 40, bins should be 40 or more.
       bins <- Array(40, 300))
    yield {
      val model = DecisionTree.trainClassifier(
        trainData, 7, Map(10 -> 4, 11 -> 40), impurity, depth, bins)
      val trainAccuracy = getMetrics(model, trainData).accuracy
      val cvAccuracy = getMetrics(model, cvData).accuracy
      ((impurity, depth, bins), (trainAccuracy, cvAccuracy))
    }

/*
    Random Forest
*/

val forest = RandomForest.trainClassifier(
  trainData, 7, Map(10 -> 4, 11 -> 40),
  20,           // # of trees
  "auto",       // Criteria on what features will be selected in each step
  "entropy",
  30,
  300)

def getRandomForestMetrics(model: RandomForestModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
  val predictionsAndLabels = data.map(example =>
    (model.predict(example.features), example.label)
  )
  // Evaluation class for quality of model
  // Refer to BinaryClassificationMetrics
  new MulticlassMetrics(predictionsAndLabels)
}
