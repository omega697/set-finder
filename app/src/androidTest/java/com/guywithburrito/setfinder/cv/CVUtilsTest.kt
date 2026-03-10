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

    @Test
    fun isWhiteCard_pureWhite_returnsTrue() {
        val quad = MatOfPoint2f(Point(10.0, 10.0), Point(90.0, 10.0), Point(90.0, 140.0), Point(10.0, 140.0))
        // Create 100x150 Lab image: L=200, a=128, b=128 (Bright neutral white)
        val lab = org.opencv.core.Mat(150, 100, org.opencv.core.CvType.CV_8UC3, org.opencv.core.Scalar(200.0, 128.0, 128.0))
        
        assertThat(quad.isWhiteCard(lab)).isTrue()
        lab.release()
    }

    @Test
    fun isWhiteCard_darkDesk_returnsFalse() {
        val quad = MatOfPoint2f(Point(10.0, 10.0), Point(90.0, 10.0), Point(90.0, 140.0), Point(10.0, 140.0))
        // Create 100x150 Lab image: L=50, a=128, b=128 (Dark grey/black)
        val lab = org.opencv.core.Mat(150, 100, org.opencv.core.CvType.CV_8UC3, org.opencv.core.Scalar(50.0, 128.0, 128.0))
        
        assertThat(quad.isWhiteCard(lab)).isFalse()
        lab.release()
    }

    @Test
    fun isWhiteCard_saturatedRed_returnsFalse() {
        val quad = MatOfPoint2f(Point(10.0, 10.0), Point(90.0, 10.0), Point(90.0, 140.0), Point(10.0, 140.0))
        // Create 100x150 Lab image: L=180 (Bright), a=200 (Strong Red), b=150 (Slight Yellow)
        // avgChrom will be sqrt((200-128)^2 + (150-128)^2) = sqrt(72^2 + 22^2) ~= 75 > 15
        val lab = org.opencv.core.Mat(150, 100, org.opencv.core.CvType.CV_8UC3, org.opencv.core.Scalar(180.0, 200.0, 150.0))
        
        assertThat(quad.isWhiteCard(lab)).isFalse()
        lab.release()
    }

    @Test
    fun isWhiteCard_offWhiteShadow_returnsTrue() {
        val quad = MatOfPoint2f(Point(10.0, 10.0), Point(90.0, 10.0), Point(90.0, 140.0), Point(10.0, 140.0))
        // L=170 (Bright but shadowed), a=132, b=135 (Slightly warm/yellowish indoor lighting)
        // Chrom distance = sqrt(4^2 + 7^2) ~= 8.06 < 15.0
        val lab = org.opencv.core.Mat(150, 100, org.opencv.core.CvType.CV_8UC3, org.opencv.core.Scalar(170.0, 132.0, 135.0))
        
        assertThat(quad.isWhiteCard(lab)).isTrue()
        lab.release()
    }
}
