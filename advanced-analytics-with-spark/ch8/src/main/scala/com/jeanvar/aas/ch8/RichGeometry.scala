package com.jeanvar.aas.ch8

import scala.language.implicitConversions
import com.esri.core.geometry.Geometry
import com.esri.core.geometry.GeometryEngine
import com.esri.core.geometry.SpatialReference

class RichGeometry(val geometry: Geometry, 
    val spatialReference: SpatialReference = SpatialReference.create(4326)) extends Serializable {

  def area2D() = geometry.calculateArea2D()

  def contains(other: Geometry): Boolean = {
    GeometryEngine.contains(geometry, other, spatialReference)
  }

  def distance(other: Geometry): Double = {
    GeometryEngine.distance(geometry, other, spatialReference)
  }
  def within(other: Geometry): Boolean = {
    GeometryEngine.within(geometry, other, spatialReference)
  }

  def overlaps(other: Geometry): Boolean = {
    GeometryEngine.overlaps(geometry, other, spatialReference)
  }

  def touches(other: Geometry): Boolean = {
    GeometryEngine.touches(geometry, other, spatialReference)
  }

  def crosses(other: Geometry): Boolean = {
    GeometryEngine.crosses(geometry, other, spatialReference)
  }

  def disjoint(other: Geometry): Boolean = {
    GeometryEngine.disjoint(geometry, other, spatialReference)
  }
}

object RichGeometry extends Serializable {
  implicit def wrapRichGeo(g: Geometry) = {
    new RichGeometry(g)
  }
}