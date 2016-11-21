import edu.umd.cloud9.collection.XMLInputFormat
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io._

val path = "file:///Users/taeksoonjang/Downloads/wikidump.xml"
@transient val conf = new Configuration()
conf.set(XMLInputFormat.START_TAG_KEY, "<page>")
conf.set(XMLInputFormat.END_TAG_KEY, "</page>")
val kvs = sc.newAPIHadoopFile(path, classOf[XMLInputFormat],
  classOf[LongWritable], classOf[Text], conf)
val rawXmls = kvs.map(p => p._2.toString)

import edu.umd.cloud9.collection.wikipedia.language._
import edu.umd.cloud9.collection.wikipedia._

def wikiXmlToPlainText(xml: String): Option[(String, String)] = {
  val page = new EnglishWikipediaPage()
  WikipediaPage.readPage(page, xml)
  if (page.isEmpty) None
  else Some((page.getTitle, page.getContent))
}

val plainText = rawXmls.flatMap(wikiXmlToPlainText)

/*
    Lemmatization
    - monkeys => monkey
    - nationalize => nationalization
    - drew => draw
*/

import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer
import java.util.Properties
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.ling.CoreAnnotations._
import org.apache.spark.rdd._

def createNLPPipeline(): StanfordCoreNLP = {
  val props = new Properties()
  props.put("annotators", "tokenize, ssplit, pos, lemma")
  new StanfordCoreNLP(props)
}

def isOnlyLetters(str: String): Boolean = {
  str.forall(c => Character.isLetter(c))
}

// stopwords : unnecessary words such as the verb to be like 'is'
def plainTextToLemmas(text: String, stopWords: Set[String],
    pipeline: StanfordCoreNLP): Seq[String] = {
  val doc = new Annotation(text)
  pipeline.annotate(doc)

  val lemmas = new ArrayBuffer[String]()
  val sentences = doc.get(classOf[SentencesAnnotation])
  for (sentence <- sentences;
       token <- sentence.get(classOf[TokensAnnotation])) {
    val lemma = token.get(classOf[LemmaAnnotation])
    if (lemma.length > 2 &&
        !stopWords.contains(lemma) &&
        isOnlyLetters(lemma)) {
      lemmas += lemma.toLowerCase
    }
  }

  lemmas
}

val stopWords = sc.broadcast(scala.io.Source.fromFile("ch6/stopwords.txt").getLines().toSet).value
val lemmatized: RDD[Seq[String]] = plainText.mapPartitions(it => {
  val pipeline = createNLPPipeline()
  it.map { case(title, contents) =>
    plainTextToLemmas(contents, stopWords, pipeline)
  }
})

/*
    Let's calculate TF-IDF
*/

import scala.collection.mutable.HashMap

val docTermFreqs = lemmatized.map(terms => {
  val termFreqs = terms.foldLeft(new HashMap[String, Int]()) {
    (map, term) => {
      map += term -> (map.getOrElse(term, 0) + 1)
      map
    }
  }
  termFreqs
})

docTermFreqs.cache()

val zero = new HashMap[String, Int]()
def merge(dfs: HashMap[String, Int], tfs: HashMap[String, Int]): HashMap[String, Int] = {
  tfs.keySet.foreach { term =>
    dfs += term -> (dfs.getOrElse(term, 0) + 1)
  }
  dfs
}

def comb(dfs1: HashMap[String, Int], dfs2: HashMap[String, Int]): HashMap[String, Int] = {
  for ((term, count) <- dfs2) {
    dfs1 += term -> (dfs1.getOrElse(term, 0) + count)
  }
  dfs1
}

// It may crash with OOM error (Java heap space)
// docTermFreqs.aggregate(zero)(merge, comb)

// It turned out that there are too many words 
// docTermFreqs.flatMap(_.keySet).distinct().count()

val docFreqs = docTermFreqs.flatMap(_.keySet).map((_, 1)).reduceByKey(_ + _)
val numTerms = 50000
val ordering = Ordering.by[(String, Int), Int](_._2)
val topDocFreqs = docFreqs.top(numTerms)(ordering)











