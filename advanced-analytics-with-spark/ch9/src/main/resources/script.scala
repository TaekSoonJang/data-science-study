import java.io.File
import java.text.SimpleDateFormat
import scala.io.Source
import com.github.nscala_time.time.Imports._

def readInvestingDotComHistory(file: File): Array[(DateTime, Double)] = {
  val format = new SimpleDateFormat("MMM d, yyyy")
  val lines = Source.fromFile(file).getLines().toSeq
  lines.map(line => {
    val cols = line.split('\t')
    val date = new DateTime(format.parse(cols(0)))
    val value = cols(1).toDouble
    (date, value)
  }).reverse.toArray
}

def readYahooHistory(file: File): Array[(DateTime, Double)] = {
  val format = new SimpleDateFormat("yyyy-MM-dd")
  val lines = Source.fromFile(file).getLines().toSeq
  lines.tail.map(line => {
    val cols = line.split(',')
    val date = new DateTime(format.parse(cols(0)))
    val value = cols(1).toDouble
    (date, value)  
  }).reverse.toArray
}

val start = new DateTime(2009, 10, 23, 0, 0)
val end = new DateTime(2014, 10, 23, 0, 0)

val files = new File("/Users/taeksoonjang/SourceTreeRepos/data-science-study/advanced-analytics-with-spark/ch9/data/stocks/").listFiles()
val rawStocks: Seq[Array[(DateTime, Double)]] = 
  files.flatMap(file => {
    try {
      Some(readYahooHistory(file))
    } catch {
      case e: Exception => None
    }
  }).filter(_.size >= 260 * 5 + 10)

val factorsPrefix = "/Users/taeksoonjang/SourceTreeRepos/data-science-study/advanced-analytics-with-spark/ch9/data/factors/"
val factors1: Seq[Array[(DateTime, Double)]] =
  Array("crudeoil.tsv", "us30yeartreasurybonds.tsv").
  map(x => new File(factorsPrefix + x)).
  map(readInvestingDotComHistory)

val factors2: Seq[Array[(DateTime, Double)]] =
  Array("SNP.csv", "NDX.csv").
  map(x => new File(factorsPrefix + x)).
  map(readYahooHistory)

def trimToRegion(history: Array[(DateTime, Double)],
    start: DateTime, end: DateTime): Array[(DateTime, Double)] = {
  var trimmed = history.dropWhile(_._1 < start).takeWhile(_._1 <= end)
  if (trimmed.head._1 != start) {
    trimmed = Array((start, trimmed.head._2)) ++ trimmed
  }
  if (trimmed.last._1 != end) {
    trimmed = trimmed ++ Array((end, trimmed.last._2))
  }

  trimmed
}

/* Interpolation for empty field */

import scala.collection.mutable.ArrayBuffer

def fillInHistory(history: Array[(DateTime, Double)],
    start: DateTime, end: DateTime): Array[(DateTime, Double)] = {
  var cur = history
  val filled = new ArrayBuffer[(DateTime, Double)]()
  var curDate = start
  while (curDate < end) {
    if (cur.tail.nonEmpty && cur.tail.head._1 == curDate) {
      cur = cur.tail
    }

    filled += ((curDate, cur.head._2))

    curDate += 1.days

    // skip weekend
    if (curDate.dayOfWeek().get > 5) curDate += 2.days
  }

  filled.toArray
}

val stocks: Seq[Array[(DateTime, Double)]] = rawStocks.
  map(trimToRegion(_, start, end)).
  map(fillInHistory(_, start, end))

val factors: Seq[Array[(DateTime, Double)]] = (factors1 ++ factors2).
  map(trimToRegion(_, start, end)).
  map(fillInHistory(_, start, end))

def twoWeekReturns(history: Array[(DateTime, Double)]): Array[Double] = {
  history.sliding(10).
    map { window =>
      val next = window.last._2
      val prev = window.head._2
      (next - prev) / prev
    }.toArray
}

val stocksReturns = stocks.map(twoWeekReturns)
val factorReturns = factors.map(twoWeekReturns)

def factorMatrix(histories: Seq[Array[Double]]): Array[Array[Double]] = {
  val mat = new Array[Array[Double]](histories.head.length)
  for (i <- 0 until histories.head.length) {
    mat(i) = histories.map(_(i)).toArray
  }
  mat
}

val factorMat = factorMatrix(factorReturns)

def featurize(factorReturns: Array[Double]): Array[Double] = {
  val squaredReturns = factorReturns.
    map(x => math.signum(x) * x * x)
  val squaredRootedReturns = factorReturns.
    map(x => math.signum(x) * math.sqrt(math.abs(x)))

  squaredReturns ++ squaredRootedReturns ++ factorReturns
}

val factorFeatures = factorMat.map(featurize)

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression

def linearModel(instrument: Array[Double],
      factorMatrix: Array[Array[Double]]): OLSMultipleLinearRegression = {
  val regression = new OLSMultipleLinearRegression()
  regression.newSampleData(instrument, factorMatrix)
  regression
}

val models = stocksReturns.map(linearModel(_, factorFeatures))

val factorWeights = models.map(_.estimateRegressionParameters()).toArray

import com.jeanvar.aas.ch9._
import breeze.plot._

def plotDistribution(samples: Array[Double]) {
  val min = samples.min
  val max = samples.max
  val domain = Range.Double(min, max, (max - min) / 100).toList.toArray
  val densities = KernelDensity.estimate(samples, domain)

  val f = Figure()
  val p = f.subplot(0)
  p += plot(domain, densities)
  p.xlabel = "Two Week Return ($)"
  p.ylabel = "Density"
}

plotDistribution(factorReturns(0))
plotDistribution(factorReturns(1))

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation

val factorCor = new PearsonsCorrelation(factorMat).getCorrelationMatrix().getData()
println(factorCor.map(_.mkString("\t")).mkString("\n"))

import org.apache.commons.math3.stat.correlation.Covariance

val factorCov = new Covariance(factorMat).getCovarianceMatrix().getData()

val factorMeans = factorReturns.map(factor => factor.sum / factor.size).toArray

import org.apache.commons.math3.distribution.MultivariateNormalDistribution

val factorsDist = new MultivariateNormalDistribution(factorMeans, factorCov)

factorsDist.sample()
factorsDist.sample()

/* Let's go to the experiments. */
val parallelism = 1000
val baseSeed = 1496

val seeds = (baseSeed until baseSeed + parallelism)
val seedRdd = sc.parallelize(seeds, parallelism)

def instrumentTrialReturn(instrument: Array[Double], trial: Array[Double]): Double = {
  val instrumentTrialReturn = instrument(0)
  var i = 0
  while(i < trial.length) {
    instrumentTrialReturn += trial(i) * instrument(i + 1)
    i += 1
  }

  instrumentTrialReturn
}

def trialReturn(trial: Array[Double], instruments: Seq[Array[Double]]): Double = {
  val totalReturn = 0.0
  for (instrument <- instruments) {
    totalReturn += instrumentTrialReturn(instrument, trial)
  }

  totalReturn / instruments.size
}

import org.apache.commons.math3.random.MersenneTwister

def trialReturns(seed: Long, numTrials: Int,
    instruments: Seq[Array[Double]], factorMeans: Array[Double],
    factorCovariances: Array[Array[Double]]): Seq[Double] = {
  val rand = new MersenneTwister(seed)
  val multivariateNormal = new MultivariateNormalDistribution(
    rand, factorMeans, factorCovariances)

  val trialReturns = new Array[Double](numTrials)
  for (i <- 0 until numTrials) {
    val trialFactorReturns = multivariateNormal.sample()
    val trialFeatures = featurize(trialFeatures, instruments)
  }

  trialReturns
}

/* stopped because I didn't understnad what is going on */







