package com.guywithburrito.setfinder.cv

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.common.truth.Truth.assertThat
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point

/**
 * Instrumented tests for geometric OpenCV utilities.
 */
@RunWith(AndroidJUnit4::class)
class CVUtilsTest {

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
    }

    @Test
    fun getCenter_calculatesCentroid() {
        val quad = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0)
        )
        val center = quad.getCenter()
        assertThat(center.x).isEqualTo(50.0)
        assertThat(center.y).isEqualTo(50.0)
    }

    @Test
    fun calculateIoU_perfectMatch_isOne() {
        val q1 = MatOfPoint2f(Point(0.0, 0.0), Point(10.0, 0.0), Point(10.0, 10.0), Point(0.0, 10.0))
        val q2 = MatOfPoint2f(Point(0.0, 0.0), Point(10.0, 0.0), Point(10.0, 10.0), Point(0.0, 10.0))
        
        assertThat(q1.calculateIoU(q2)).isWithin(0.01).of(1.0)
    }

    @Test
    fun calculateIoU_halfOverlap_isCorrect() {
        // 10x10 square at (0,0) -> Area 100
        val q1 = MatOfPoint2f(Point(0.0, 0.0), Point(10.0, 0.0), Point(10.0, 10.0), Point(0.0, 10.0))
        // 10x10 square at (5,0) -> Area 100, Intersect at (5,0) with width 5 -> Area 50
        val q2 = MatOfPoint2f(Point(5.0, 0.0), Point(15.0, 0.0), Point(15.0, 10.0), Point(5.0, 10.0))
        
        // IoU = 50 / (100 + 100 - 50) = 50 / 150 = 0.333
        assertThat(q1.calculateIoU(q2)).isWithin(0.01).of(0.333)
    }

    @Test
    fun calculateIoU_noOverlap_isZero() {
        val q1 = MatOfPoint2f(Point(0.0, 0.0), Point(10.0, 0.0), Point(10.0, 10.0), Point(0.0, 10.0))
        val q2 = MatOfPoint2f(Point(20.0, 20.0), Point(30.0, 20.0), Point(30.0, 30.0), Point(20.0, 30.0))
        
        assertThat(q1.calculateIoU(q2)).isEqualTo(0.0)
    }
}
