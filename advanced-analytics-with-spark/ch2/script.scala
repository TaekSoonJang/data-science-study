// Read all textfile in the directory
val rawblocks = sc.textFile("/Users/taeksoonjang/Downloads/linkage/csv")

def isHeader(line: String) = {
	line.contains("id_1")
}

val noheader = rawblocks.filter(x => !isHeader(x))

def toDouble(s: String) = {
  if ("?".equals(s)) Double.NaN else s.toDouble
}

case class MatchData(id1: Int, id2: Int, scores: Array[Double], matched: Boolean)

def parse(line: String) = {
  val pieces = line.split(',')
  val id1 = pieces(0).toInt
  val id2 = pieces(1).toInt
  val scores = pieces.slice(2, 11).map(toDouble)
  val matched = pieces(11).toBoolean
  MatchData(id1, id2, scores, matched)
}

val parsed = noheader.map(line => parse(line))

// PairRDD.countByValue() : groupBy value with its count
val matchCounts = parsed.map(md => md.matched).countByValue()
val matchCountsSeq = matchCounts.toSeq
matchCountsSeq.sortBy(_._1).foreach(println)
matchCountsSeq.sortBy(_._2).foreach(println)

// It's okay but inefficient to iterate 10 times for all values
import java.lang.Double.isNaN
val stats = (0 until 9).map(i => {
  // filter NaN value
	parsed.map(md => md.scores(i)).filter(!isNaN(_)).stats()
})

/*
    Efficient way to handle all double values including missing values

    To use function 'statsWithMissing', load StatsWithMissing.scala in spark shell as below:
    spark> :load /path/to/StatsWithMissing.scala
*/

val statsm = statsWithMissing(parsed.filter(_.matched).map(_.scores))
val statsn = statsWithMissing(parsed.filter(!_.matched).map(_.scores))

// Find out which features are good
// Good features should have little missing values and difference between comparison groups should be high
statsm.zip(statsn).map { case (m, n) => {
  (m.missing + n.missing, m.stats.mean - n.stats.mean)
}}.foreach(println)

// After found good features, let's make a model
def naz(d: Double) = if (Double.NaN.equals(d)) 0.0 else d
case class Scored(md: MatchData, score: Double)
val ct = parsed.map(md => {
  val score = Array(2, 5, 6, 7, 8).map(i => naz(md.scores(i))).sum
  Scored(md, score)
})

ct.filter(s => s.score >= 4.0).map(s => s.md.matched).countByValue()