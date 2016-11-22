package com.jeanvar.aas.ch8

import com.esri.core.geometry.{Geometry, GeometryEngine}
import org.apache.spark.{HashPartitioner, Partitioner}
import spray.json._

case class Feature(
    val id: Option[JsValue],
    val properties: Map[String, JsValue],
    val geometry: RichGeometry) {
  def apply(property: String) = properties(property)
  def get(property: String) = properties.get(property)
}

import com.esri.core.geometry.Point
import com.github.nscala_time.time.Imports._

case class Trip(
     pickupTime: DateTime,
     dropoffTime: DateTime,
     pickupLoc: Point,
     dropoffLoc: Point)

case class FeatureCollection(features: Array[Feature]) extends IndexedSeq[Feature] {
  def apply(index: Int) = features(index)
  def length = features.length
}

case class GeometryCollection(geometries: Array[RichGeometry])
  extends IndexedSeq[RichGeometry] {
  def apply(index: Int) = geometries(index)
  def length = geometries.length
}

object GeoJsonProtocol extends DefaultJsonProtocol {
  implicit object RichGeometryJsonFormat extends RootJsonFormat[RichGeometry] {
    def write(g: RichGeometry) = {
      GeometryEngine.geometryToGeoJson(g.spatialReference, g.geometry).parseJson
    }
    def read(value: JsValue) = {
      val mg = GeometryEngine.geometryFromGeoJson(value.compactPrint, 0, Geometry.Type.Unknown)
      new RichGeometry(mg.getGeometry, mg.getSpatialReference)
    }
  }

  implicit object FeatureJsonFormat extends RootJsonFormat[Feature] {
    def write(f: Feature) = {
      val buf = scala.collection.mutable.ArrayBuffer(
        "type" -> JsString("Feature"),
        "properties" -> JsObject(f.properties),
        "geometry" -> f.geometry.toJson)
      f.id.foreach(v => { buf += "id" -> v})
      JsObject(buf.toMap)
    }

    def read(value: JsValue) = {
      val jso = value.asJsObject
      val id = jso.fields.get("id")
      val properties = jso.fields("properties").asJsObject.fields
      val geometry = jso.fields("geometry").convertTo[RichGeometry]
      Feature(id, properties, geometry)
    }
  }

  implicit object FeatureCollectionJsonFormat extends RootJsonFormat[FeatureCollection] {
    def write(fc: FeatureCollection) = {
      JsObject(
        "type" -> JsString("FeatureCollection"),
        "features" -> JsArray(fc.features.map(_.toJson): _*)
      )
    }

    def read(value: JsValue) = {
      FeatureCollection(value.asJsObject.fields("features").convertTo[Array[Feature]])
    }
  }

  implicit object GeometryCollectionJsonFormat extends RootJsonFormat[GeometryCollection] {
    def write(gc: GeometryCollection) = {
      JsObject(
        "type" -> JsString("GeometryCollection"),
        "geometries" -> JsArray(gc.geometries.map(_.toJson): _*))
    }

    def read(value: JsValue) = {
      GeometryCollection(value.asJsObject.fields("geometries").convertTo[Array[RichGeometry]])
    }
  }
}

class FirstKeyPartitioner[K1, K2](partitions: Int) extends Partitioner {
  val delegate = new HashPartitioner(partitions)
  override def numPartitions = delegate.numPartitions
  override def getPartition(key: Any): Int = {
    val k = key.asInstanceOf[(K1, K2)]
    delegate.getPartition(k._1)
  }
}

