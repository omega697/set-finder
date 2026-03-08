package com.guywithburrito.setfinder

import com.guywithburrito.setfinder.tracking.CardTracker
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.opencv.core.Point

class CardTrackerTest {

    private val boundsA = listOf(Point(0.0, 0.0), Point(10.0, 0.0), Point(10.0, 10.0), Point(0.0, 10.0))

    @Test
    fun tracker_returnsCardsImmediately() {
        val tracker = CardTracker()
        
        // 1st frame - Should return tracked cards immediately with framesSeen=1
        val tracked = tracker.updateGeometric(listOf(boundsA))
        assertThat(tracked).hasSize(1)
        assertThat(tracked[0].framesSeen).isEqualTo(1)
    }

    @Test
    fun tracker_matchesProximity() {
        val tracker = CardTracker()
        
        // Initial detection
        tracker.updateGeometric(listOf(boundsA))

        // Second detection slightly offset (well within 80px threshold)
        val boundsB = listOf(Point(5.0, 5.0), Point(15.0, 5.0), Point(15.0, 15.0), Point(5.0, 15.0))
        val tracked = tracker.updateGeometric(listOf(boundsB))

        assertThat(tracked).hasSize(1)
        assertThat(tracked[0].framesSeen).isEqualTo(2)
        // Bounds should be smoothed: 0.5 * 0.0 + 0.5 * 5.0 = 2.5
        assertThat(tracked[0].bounds[0].x).isEqualTo(2.5)
    }

    @Test
    fun tracker_expiresAfterTimeout() {
        val tracker = CardTracker()
        
        // Initial detection
        tracker.updateGeometric(listOf(boundsA))
        
        // Simulate time passing (EXPIRY_MS is 600)
        Thread.sleep(700)
        
        val tracked = tracker.updateGeometric(emptyList())
        assertThat(tracked).isEmpty()
    }
}
